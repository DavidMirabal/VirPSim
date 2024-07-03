import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np


def generar_colores(N):
    valores = np.linspace(0, 1, N)
    colores = plt.cm.rainbow(valores)
    colores_hex = [to_hex(color) for color in colores]
    return colores_hex


class Figura:
    def __init__(
        self,
        ancho=6,
        ratio=16 / 9,
        dpi=500,
        c_text="white",
        c_face="black",
        c_axe="black",
        c_spine="white",
        ticks=("no", "no"),
        lw_spine=0,
        s_text=8,
        fontsize=18,
    ):
        plt.style.use(['seaborn-v0_8-colorblind', "D:/plstyle_dmb2.mplstyle"])

        fig = plt.figure(figsize=(ancho, ancho / (ratio)), dpi=dpi)

        fig.set_facecolor(c_face)

        self.fig = fig
        self.dpi = dpi
        self.ax = []
        self.ticks = ticks
        self.c_text = c_text
        self.c_face = c_face
        self.c_axe = c_axe
        self.c_spine = c_spine

        self.lw_spine = lw_spine
        self.s_text = s_text
        self.fontsize = fontsize

    def axs(self, ncols=1, nrows=1, *args):
        k = 0
        if len(args) > 0:
            projection = args[0]
        for i in range(ncols):
            for j in range(nrows):
                if projection[k] == '3d':
                    self.ax.append(
                        self.fig.add_subplot(
                            nrows, ncols, k + 1, projection=projection[k]
                        )
                    )
                else:
                    self.ax.append(self.fig.add_subplot(nrows, ncols, k + 1))

                for spine in self.ax[k].spines.values():
                    spine.set_linewidth(self.lw_spine)
                    spine.set_color(self.c_spine)

                self.ax[-1].set_facecolor(self.c_axe)

                if self.ticks[0] == "no":
                    self.ax[-1].set_xticks([])

                if self.ticks[1] == "no":
                    self.ax[-1].set_yticks([])
                else:
                    self.ax[-1].tick_params(
                        axis='both', colors=self.c_text, labelsize=self.s_text
                    )

                k = k + 1

        return self.fig, self.ax

    def xlabel(self, index_ax=-1, s=r""):
        self.ax[index_ax].set_xlabel(
            r"$" + s + r"$", color=self.c_text, fontsize=self.fontsize
        )

    def ylabel(self, index_ax=-1, s=r""):
        self.ax[index_ax].set_ylabel(
            r"$" + s + r"$", color=self.c_text, fontsize=self.fontsize
        )
