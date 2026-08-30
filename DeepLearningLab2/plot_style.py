import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Times New Roman on Colab comes from the msttcorefonts package.
# Run this in a Colab cell once, before running any of the numbered
# scripts below, to install it:
#
#   !echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
#   !sudo apt-get install -y ttf-mscorefonts-installer
#   !sudo fc-cache -f
#
FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

FIG_FORMAT = "eps"
FIG_DPI = 600


def setup_fonts():
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = "Times New Roman"
    except FileNotFoundError:
        print("Times New Roman not found at", FONT_PATH)
        print("Run the msttcorefonts install cell in Colab first (see plot_style.py).")
        print("Falling back to a generic serif font for now.")
        plt.rcParams["font.family"] = "serif"


def save_fig(fig, path_no_ext):
    out_path = f"{path_no_ext}.{FIG_FORMAT}"
    fig.savefig(out_path, format=FIG_FORMAT, dpi=FIG_DPI, bbox_inches="tight")
    return out_path
