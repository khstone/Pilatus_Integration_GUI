"""Add red numbered callouts (style of the original manual) to screenshots."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
C = json.load(open(os.path.join(D, "callouts.json")))
RED = "#c8102e"


def annotate(name, marks, right=(), top=(), out=None, margin=70):
    img = mpimg.imread(os.path.join(D, name + ".png"))
    h, w = img.shape[:2]
    W = w + 2 * margin
    fig = plt.figure(figsize=(W / 100, (h + margin) / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(h + margin, 0); ax.axis("off")
    ax.imshow(img, extent=[margin, margin + w, h + margin, margin])
    ax.add_patch(plt.Rectangle((margin, margin), w, h, fill=False, ec="#8a8984", lw=0.8))
    for k, (x, y, rw, rh) in marks.items():
        x, y = x + margin, y + margin
        cy = y + rh / 2
        if k in top:
            tx, ty = x + 12, margin * 0.45
            ax.annotate(f"{k}.", xy=(x + 12, y + 4), xytext=(tx, ty), color=RED, fontsize=15,
                        fontweight="bold", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=RED, lw=1.6))
        elif k in right:
            ax.annotate(f"{k}.", xy=(x + rw - 4, cy), xytext=(W - margin * 0.45, cy), color=RED, fontsize=15,
                        fontweight="bold", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=RED, lw=1.6))
        else:
            # stop at the window edge: point at the row without striking through its label
            ax.annotate(f"{k}.", xy=(margin + 6, cy), xytext=(margin * 0.45, cy), color=RED, fontsize=15,
                        fontweight="bold", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=RED, lw=1.6))
    fig.savefig(os.path.join(D, (out or name) + "_annotated.png"), dpi=100)
    plt.close(fig)


L = C["01_layout"]
# Integrate (8) and Overlay/Contour (9) share a row: one callout; renumber the rest.
layout = {"1": L["1"], "2": L["2"], "3": L["3"], "4": L["4"], "5": L["5"], "6": L["6"], "7": L["7"],
          "8": [L["8"][0], L["8"][1], L["9"][0] + L["9"][2] - L["8"][0], L["8"][3]],
          "9": L["10"], "10": L["11"], "11": L["12"], "12": L["13"], "13": L["14"]}
annotate("01_layout", layout, right=("11", "12", "13"), top=("1",))
V = C["06_live"]
# One callout per row: (live toggle + start at), (detect + waterfall), status, events, plot
live = {"1": V["1"], "2": V["3"], "3": V["5"], "4": V["6"], "5": V["7"]}
annotate("06_live", live, right=("5",))
print("ok")
