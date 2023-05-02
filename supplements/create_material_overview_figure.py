import json
import numpy as np
import matplotlib.pyplot as plt

ALL_PHANTOMS = ['P.1.1.a', 'P.1.11', 'P.1.2.a', 'P.1.3.a', 'P.1.6', 'P.3.10', 'P.3.11', 'P.3.12', 'P.3.13', 'P.3.14', 'P.3.15', 'P.3.16', 'P.3.17', 'P.3.18', 'P.3.19', 'P.3.1', 'P.3.20', 'P.3.21', 'P.3.22', 'P.3.23', 'P.3.24', 'P.3.25', 'P.3.26', 'P.3.27', 'P.3.28', 'P.3.2', 'P.3.30', 'P.3.31', 'P.3.32', 'P.3.35', 'P.3.36', 'P.3.37', 'P.3.38', 'P.3.3', 'P.3.40', 'P.3.41', 'P.3.44', 'P.3.46', 'P.3.47', 'P.3.48', 'P.3.4', 'P.3.50', 'P.3.55', 'P.3.57', 'P.3.59', 'P.3.5', 'P.3.60', 'P.3.61', 'P.3.62', 'P.3.64', 'P.3.65', 'P.3.67', 'P.3.68', 'P.3.69', 'P.3.6', 'P.3.71', 'P.3.72', 'P.3.7', 'P.3.8', 'P.3.9', 'P.5.10.2', 'P.5.10.3', 'P.5.11', 'P.5.12', 'P.5.19', 'P.5.1', 'P.5.2.2', 'P.5.2.3', 'P.5.20', 'P.5.25', 'P.5.27', 'P.5.29', 'P.5.31', 'P.5.32', 'P.5.3', 'P.5.4.2', 'P.5.4.3', 'P.5.5', 'P.5.6.2', 'P.5.6.3', 'P.5.7', 'P.5.8.2', 'P.5.8.3', 'P.5.9', 'P.2.1', 'P.2.2', 'P.2.3', 'P.2.4', 'P.3.34', 'P.3.52', 'P.3.53', 'P.3.74', 'P.3.76', 'P.3.78', 'P.5.13', 'P.5.14', 'P.5.15', 'P.5.16', 'P.5.21', 'P.5.22', 'P.5.23', 'P.5.24']
INPUT_PATH = r"H:\DIS/"
WAVELENGTHS = np.arange(700, 901, 10)

with open(INPUT_PATH + "/material_mapping.json", "r+") as material_mapping_file:
    material_mapping = json.load(material_mapping_file)

phantom_names = list(material_mapping.keys())[::-1]
background_muas = []
background_musps = []
inclusion_muas = []
inclusion_musps = []

table_string = (r"\begin{longtable}[h!tb]{cccc}"
                "\n"
                r"\caption{Tabulated overview of the characteristic optical properties of the tissue-mimicking phantoms at the reference wavelength of 800\,nm. Shown are the median results $\pm$ the standard deviations over the 10 double-integrating sphere measurements made for each sample.}"
                "\n"
                r"\label{tab:my_label}\\"
                "\n"
                r"\textbf{Phantom Identifier} & \textbf{Region} & \textbf{Absorption Coefficient [cm$^{-1}$]} & \textbf{Reduced Scattering Coefficient [cm$^{-1}$]}\\"
                "\n"
                r"\hline"
                "\n"
                r"\\"
                "\n")

for phantom in phantom_names:
    if phantom in ALL_PHANTOMS:
        phantom_data = material_mapping[phantom]
        segmentation_regions = list(phantom_data.keys())
        bg_key = phantom_data["1"]
        if bg_key[-1] == "a":
            background_properties = np.load(INPUT_PATH + "/processed/" + bg_key + "/" + bg_key[:-2] + ".npz")
        else:
            background_properties = np.load(INPUT_PATH + "/processed/" + bg_key + ".a/" + bg_key + ".npz")

        wl = background_properties["wavelengths"]
        mua = background_properties["mua"]
        mua_std = background_properties["mua_std"]
        musp = background_properties["mus"] * 0.3
        musp_std = background_properties["mus_std"] * 0.3

        for lamda in WAVELENGTHS:
            background_muas.append(mua[np.argwhere(wl == lamda)].item())
            background_musps.append(musp[np.argwhere(wl == lamda)].item())

        ref_mua = mua[np.argwhere(wl == 800)].item()
        ref_mua_std = mua_std[np.argwhere(wl == 800)].item()
        ref_mus = musp[np.argwhere(wl == 800)].item()
        ref_mus_std = musp_std[np.argwhere(wl == 800)].item()

        table_string += rf"\textbf{{{phantom}}} & Background & {ref_mua:.2f} $\pm$ {ref_mua_std:.3f} & {ref_mus:.2f} $\pm$ {ref_mus_std:.2f} \\"
        table_string += "\n"

        segmentation_regions = segmentation_regions[2:]

        for reg in segmentation_regions:

            bg_key = phantom_data[reg]
            if bg_key[-1] == "a":
                background_properties = np.load(INPUT_PATH + "/processed/" + bg_key + "/" + bg_key[:-2] + ".npz")
            else:
                background_properties = np.load(INPUT_PATH + "/processed/" + bg_key + ".a/" + bg_key + ".npz")

            wl = background_properties["wavelengths"]
            mua = background_properties["mua"]
            mua_std = background_properties["mua_std"]
            musp = background_properties["mus"] * 0.3
            musp_std = background_properties["mus_std"] * 0.3

            for lamda in WAVELENGTHS:
                inclusion_muas.append(mua[np.argwhere(wl == lamda)].item())
                inclusion_musps.append(musp[np.argwhere(wl == lamda)].item())

            ref_mua = mua[np.argwhere(wl == 800)].item()
            ref_mua_std = mua_std[np.argwhere(wl == 800)].item()
            ref_mus = musp[np.argwhere(wl == 800)].item()
            ref_mus_std = musp_std[np.argwhere(wl == 800)].item()

            table_string += rf" & Inclusion {int(reg)-1} & {ref_mua:.2f} $\pm$ {ref_mua_std:.3f} & {ref_mus:.2f} $\pm$ {ref_mus_std:.2f} \\"
            table_string += "\n"

plt.figure(figsize=(12, 7))
N_BINS = 30
plt.subplot(2, 2, 1)
plt.title(r"Background $\mu_a$ distribution")
plt.xlabel("Optical absorption coefficient [cm$^{-1}$")
plt.ylabel("Density")
plt.text(-0.14, 1.02, "A", transform=plt.gca().transAxes,
                size=30, weight='bold')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hist(background_muas, bins=N_BINS, density=True)

plt.subplot(2, 2, 2)
plt.title(r"Inclusion $\mu_a$ distribution")
plt.xlabel("Optical absorption coefficient [cm$^{-1}$")
plt.ylabel("Density")
plt.text(-0.14, 1.02, "B", transform=plt.gca().transAxes,
                size=30, weight='bold')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hist(inclusion_muas, bins=N_BINS, density=True)

plt.subplot(2, 2, 3)
plt.title(r"Background $\mu_s'$ distribution")
plt.xlabel("Optical scattering coefficient [cm$^{-1}$")
plt.ylabel("Density")
plt.text(-0.14, 1.02, "C", transform=plt.gca().transAxes,
                size=30, weight='bold')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hist(background_musps, bins=N_BINS, density=True)

plt.subplot(2, 2, 4)
plt.title(r"Inclusion $\mu_s'$ distribution")
plt.xlabel("Optical scattering coefficient [cm$^{-1}$")
plt.ylabel("Density")
plt.text(-0.14, 1.02, "D", transform=plt.gca().transAxes,
                size=30, weight='bold')
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.hist(inclusion_musps, bins=N_BINS, density=True)

# print("bg mua", np.percentile(background_muas, 10), np.percentile(background_muas, 90))
# print("bg musp", np.percentile(background_musps, 10), np.percentile(background_musps, 90))
# print("inc mua", np.percentile(inclusion_muas, 10), np.percentile(inclusion_muas, 90))
# print("inc musp", np.percentile(inclusion_musps, 10), np.percentile(inclusion_musps, 90))

plt.tight_layout()
plt.savefig("optical_property_densities.png", dpi=300)

table_string += (
    r"\end{longtable}"
)

print(table_string)