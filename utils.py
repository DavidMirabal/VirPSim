import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
import dmb_figures as dmb
from image_to_video import ImageToVideoConverter


class simulation:
    def __init__(
        self,
        N,
        v_mean,
        per_healthy,
        per_sick,
        prob_sick,
        prob_imm_imm,
        time_die,
        radius,
        scale,
        dt,
        device,
        frames,
        dir_images,
    ):
        self.device = device
        self.radius = radius
        self.scale = scale
        self.dt = dt
        self.N = N
        self.frames = frames
        self.dir_images = dir_images
        self.prob_imm_imm = prob_imm_imm
        self.prob_imm_sick = 1 - prob_imm_imm
        self.time_die = time_die

        self.prob_sick = prob_sick
        self.prob_healthy = 1 - prob_sick

        offset = radius * 2
        self.pos = (torch.rand(N, 2) * (scale - 2 * offset) + offset).to(device)

        self.stat = torch.zeros(N).long().to(device)

        N_healthy = int(N * per_healthy)
        N_sick = int(N * per_sick)

        self.stat[:N_healthy] = 0
        self.stat[N_healthy : N_healthy + N_sick] = 1
        self.stat[N_healthy + N_sick :] = 2

        self.ids = torch.arange(N).to(device)
        self.id_pairs = torch.combinations(self.ids, 2).long().to(device)

        theta = torch.rand(N) * 2 * torch.pi
        self.v = torch.empty((N, 2)).to(device)
        self.v[:, 0] = v_mean * torch.cos(theta)
        self.v[:, 1] = v_mean * torch.sin(theta)

        self.sick_count = torch.empty(self.frames).fill_(float('nan')).cpu()
        self.healthy_count = torch.empty(self.frames).fill_(float('nan')).cpu()
        self.immune_count = torch.empty(self.frames).fill_(float('nan')).cpu()
        self.dead_count = torch.empty(self.frames).fill_(float('nan')).cpu()
        self.t = torch.arange(self.frames) * self.dt
        self.time_sick = torch.zeros(N).to(device)

        FIGURA = dmb.Figura(
            ancho=8, ticks=('yes', 'yes'), lw_spine=3, ratio=1, fontsize=20, s_text=20
        )
        self.fig, self.ax = FIGURA.axs(1, 2, ('2d', '2d'))
        self.markersize = (
            2
            * self.radius
            * self.ax[0].get_window_extent().width
            / (self.scale)
            * 72.0
            / self.fig.dpi
        )

        self.ax[0].set_xlim(0, self.scale)
        self.ax[0].set_ylim(0, self.scale)
        self.ax[1].set_xlim(0, self.t[-1])
        self.ax[1].set_ylim(0, 1)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])

        self.ax[0].set_position([0.2, 0.01, 0.7, 0.7])
        self.ax[1].set_position([0.1, 0.8, 0.8, 0.15])
        plt.close()

    def delta_pos(self):
        dx_pairs = (
            self.pos[self.id_pairs[:, 0], 0] - self.pos[self.id_pairs[:, 1], 0]
        ).to(self.device)
        dy_pairs = (
            self.pos[self.id_pairs[:, 0], 1] - self.pos[self.id_pairs[:, 1], 1]
        ).to(self.device)
        self.dpos = torch.sqrt(dx_pairs**2 + dy_pairs**2)

    def near(self):
        self.ids_col = (self.dpos < self.radius) & (
            torch.sum(
                (self.v[self.id_pairs[:, 0], :] - self.v[self.id_pairs[:, 1], :])
                * (self.pos[self.id_pairs[:, 0], :] - self.pos[self.id_pairs[:, 1], :]),
                axis=1,
            )
            < 0
        )

    def new_v(self, v1, v2, r1, r2):
        v1new = v1 - (
            torch.sum((v1 - v2) * (r1 - r2), axis=1) / torch.sum((r1 - r2) ** 2, axis=1)
        ).unsqueeze(1) * (r1 - r2)
        v2new = v2 - (
            torch.sum((v2 - v1) * (r2 - r1), axis=1) / torch.sum((r2 - r1) ** 2, axis=1)
        ).unsqueeze(1) * (r2 - r1)
        return v1new, v2new

    def health_sick(self):
        sicks = self.stat == 1
        immunes = self.stat == 2
        stat_pairs = torch.combinations(self.stat, 2).to(self.device)

        sick_p = torch.logical_or(stat_pairs[:, 0] == 1, stat_pairs[:, 1] == 1)
        new_sicks = self.id_pairs[(sick_p & self.ids_col)].unique(dim=0).flatten()

        if len(new_sicks) > 0:
            probabilities_sick = torch.tensor([self.prob_healthy, self.prob_sick])
            choices_sick = torch.multinomial(
                probabilities_sick, len(new_sicks), replacement=True
            )
            self.stat[new_sicks] = choices_sick.to(self.device)

        self.stat[sicks] = 1
        self.stat[
            self.time_sick
            >= self.time_die
            + torch.normal(0, self.time_die / 10, size=(self.N,), device=self.device)
        ] = 3

        probabilities_imm = torch.tensor([self.prob_imm_sick, self.prob_imm_imm])
        try:
            choices_imm = torch.multinomial(
                probabilities_imm, len(self.stat[self.stat == 1]), replacement=True
            )
            self.stat[self.stat == 1] = torch.tensor([1, 2])[choices_imm].to(
                self.device
            )
        except:
            pass

        self.stat[immunes] = 2
        self.id_healthy = self.ids[self.stat == 0]
        self.id_sick = self.ids[self.stat == 1]
        self.id_immune = self.ids[self.stat == 2]
        self.id_dead = self.ids[self.stat == 3]

    def motion(self):

        index0 = self.id_pairs[self.ids_col, 0]
        index1 = self.id_pairs[self.ids_col, 1]

        self.v[index0, :], self.v[index1, :] = self.new_v(
            self.v[index0, :],
            self.v[index1, :],
            self.pos[index0, :],
            self.pos[index1, :],
        )

        self.v[self.pos[:, 0] + self.radius / 2 > self.scale, 0] = -self.v[
            self.pos[:, 0] + self.radius / 2 > self.scale, 0
        ]
        self.v[self.pos[:, 1] + self.radius / 2 > self.scale, 1] = -self.v[
            self.pos[:, 1] + self.radius / 2 > self.scale, 1
        ]
        self.v[self.pos[:, 0] - self.radius / 2 < 0, 0] = -self.v[
            self.pos[:, 0] - self.radius / 2 < 0, 0
        ]
        self.v[self.pos[:, 1] - self.radius / 2 < 0, 1] = -self.v[
            self.pos[:, 1] - self.radius / 2 < 0, 1
        ]

        self.pos = self.pos + self.dt * self.v

    def run_simulation(self):
        file_list = os.listdir(self.dir_images)
        for file_name in file_list:
            file_path = os.path.join(self.dir_images, file_name)
            os.remove(file_path)
        t_show = 0
        for i in tqdm(range(self.frames), desc="Progreso"):
            self.delta_pos()
            self.near()

            self.health_sick()
            self.pos[self.id_dead, :] = -10
            self.v[self.id_dead, :] = 0

            self.sick_count[i] = len(self.id_sick) / self.N
            self.healthy_count[i] = len(self.id_healthy) / self.N
            self.immune_count[i] = len(self.id_immune) / self.N
            self.dead_count[i] = (
                1 - self.sick_count[i] - self.healthy_count[i] - self.immune_count[i]
            )

            self.ax[0].plot(
                self.pos[:, 0][self.id_healthy].cpu(),
                self.pos[:, 1][self.id_healthy].cpu(),
                color='blue',
                marker='.',
                ls='',
                markersize=self.markersize,
            )
            self.ax[0].plot(
                self.pos[:, 0][self.id_sick].cpu(),
                self.pos[:, 1][self.id_sick].cpu(),
                color='red',
                marker='.',
                ls='',
                markersize=self.markersize,
            )
            self.ax[0].plot(
                self.pos[:, 0][self.id_immune].cpu(),
                self.pos[:, 1][self.id_immune].cpu(),
                color='green',
                marker='.',
                ls='',
                markersize=self.markersize,
            )

            self.ax[1].fill_between(self.t, self.healthy_count, color='blue', alpha=0.4)
            self.ax[1].plot(
                self.t, self.healthy_count, color='blue', lw=2, label='Vulnerable'
            )
            self.ax[1].fill_between(self.t, self.sick_count, color='red', alpha=0.4)
            self.ax[1].plot(self.t, self.sick_count, color='red', lw=2, label='Sick')
            self.ax[1].fill_between(self.t, self.dead_count, color='fuchsia', alpha=0.4)
            self.ax[1].plot(
                self.t, self.dead_count, color='fuchsia', lw=2, label='Dead'
            )
            self.ax[1].fill_between(self.t, self.immune_count, color='green', alpha=0.5)
            self.ax[1].plot(
                self.t, self.immune_count, color='green', lw=2, label='Immune'
            )

            self.ax[1].legend(
                loc='lower left',
                bbox_to_anchor=(-0.12, -1.8),
                facecolor='black',
                edgecolor='white',
                framealpha=1,
                fontsize=12,
                markerscale=1,
                frameon=True,
                fancybox=False,
                labelcolor='white',
                title=f'{self.N} particles\n$t={t_show: .2f}$',
                title_fontsize=12,
            )

            self.ax[1].text(
                -0.185,
                -2.5,
                f'{len(self.id_healthy)} vulnerable',
                color='blue',
                fontsize=12,
                bbox=dict(
                    facecolor='black', edgecolor='white', boxstyle='square,pad=0.5'
                ),
                ha='left',
            )
            self.ax[1].text(
                -0.185,
                -3,
                f'{len(self.id_sick)} sick',
                color='red',
                fontsize=12,
                bbox=dict(
                    facecolor='black', edgecolor='white', boxstyle='square,pad=0.5'
                ),
                ha='left',
            )
            self.ax[1].text(
                -0.185,
                -3.5,
                f'{len(self.id_immune)} immune',
                color='green',
                fontsize=12,
                bbox=dict(
                    facecolor='black', edgecolor='white', boxstyle='square,pad=0.5'
                ),
                ha='left',
            )
            self.ax[1].text(
                -0.185,
                -4,
                f'{len(self.id_dead)} dead',
                color='fuchsia',
                fontsize=12,
                bbox=dict(
                    facecolor='black', edgecolor='white', boxstyle='square,pad=0.5'
                ),
                ha='left',
            )

            self.ax[1].get_legend().get_title().set_color('white')

            self.ax[1].set_xlabel(r'$t$', fontsize=22, c='white')

            self.ax[0].set_xlim(0, self.scale)
            self.ax[0].set_ylim(0, self.scale)
            self.ax[1].set_xlim(0, self.t[-1])
            self.ax[1].set_ylim(0, 1)
            self.ax[0].set_xticks([])
            self.ax[0].set_yticks([])

            self.fig.savefig(f'{self.dir_images}/im_{i}.jpg')

            plt.close()
            self.ax[0].cla()
            self.ax[1].cla()

            self.motion()
            t_show = t_show + self.dt
            self.time_sick[self.id_sick] = self.time_sick[self.id_sick] + self.dt
            self.time_sick[~self.id_sick] = 0

    def save_data(self, name):
        data = np.column_stack(
            (
                self.t,
                self.sick_count,
                self.healthy_count,
                self.immune_count,
                self.dead_count,
            )
        )
        np.savetxt(
            name,
            data,
            fmt='%.3f',
            delimiter='\t',
            header='t\tsick_count\thealthy_count\timmune_count\tdead_count',
        )

    def make_video(self, title, fps=30, resize=1):
        ImageToVideoConverter.png_to_mp4(
            self.dir_images,
            extension=".jpg",
            digit_format="01d",
            fps=fps,
            title=title,
            resize_factor=resize,
        )

        print('Done!')
