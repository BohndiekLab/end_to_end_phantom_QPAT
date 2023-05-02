from simpa.utils import Tags
import simpa as sp
import numpy as np
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WAVELENGTH = 800


def simulate_phantom(name, mua, mus):

    print(name)
    VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 90
    VOLUME_PLANAR_DIM_IN_MM = 30
    SPACING = 0.5
    RANDOM_SEED = 471
    SPEED_OF_SOUND = 1470

    path_manager = sp.PathManager()

    phantom_material = sp.Molecule(volume_fraction=1.0,
       absorption_spectrum=sp.AbsorptionSpectrumLibrary.CONSTANT_ABSORBER_ARBITRARY(mua.item()),
       scattering_spectrum=sp.ScatteringSpectrumLibrary.CONSTANT_SCATTERING_ARBITRARY(mus.item()),
       anisotropy_spectrum=sp.AnisotropySpectrumLibrary.CONSTANT_ANISOTROPY_ARBITRARY(0.7),
       speed_of_sound=SPEED_OF_SOUND,
       alpha_coefficient=0.001,
       density=((0.87 * 0.711 + 0.91 * 0.214 + 1.05 * 0.018) / (0.711 + 0.214 + 0.018)) * 1000,
       gruneisen_parameter=1.0,
       name="Phantom")

    def create_example_tissue():
        """
        This is a very simple example script of how to create a tissue definition.
        It contains a muscular background, an epidermis layer on top of the muscles
        and a blood vessel.
        """
        background_dictionary = sp.Settings()
        background_dictionary[Tags.MOLECULE_COMPOSITION] = (sp.MolecularCompositionGenerator()
                                                            .append(sp.MoleculeLibrary().water())
                                                            .get_molecular_composition(segmentation_type=-1))
        background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

        phantom_material_dictionary = sp.Settings()
        phantom_material_dictionary[Tags.PRIORITY] = 3
        phantom_material_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                                0,
                                                                VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
        phantom_material_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                              VOLUME_PLANAR_DIM_IN_MM,
                                                              VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
        phantom_material_dictionary[Tags.STRUCTURE_RADIUS_MM] = 13.75
        phantom_material_dictionary[Tags.MOLECULE_COMPOSITION] = (sp.MolecularCompositionGenerator()
                                                                  .append(phantom_material)
                                                                  .get_molecular_composition(segmentation_type=0))
        phantom_material_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
        phantom_material_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

        tissue_dict = sp.Settings()
        tissue_dict[Tags.BACKGROUND] = background_dictionary
        tissue_dict["phantom"] = phantom_material_dictionary
        return tissue_dict

    # Seed the numpy random configuration prior to creating the global_settings file in
    # order to ensure that the same volume
    # is generated with the same random seed every time.


    np.random.seed(RANDOM_SEED)

    settings = {
        # These parameters set he general propeties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: name + str(RANDOM_SEED),
        Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
        Tags.SPACING_MM: SPACING,
        Tags.WAVELENGTHS: [800],
        Tags.DIM_VOLUME_Z_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True
    }

    settings = sp.Settings(settings)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue(),
        Tags.SIMULATE_DEFORMED_LAYERS: False
    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_INVISION,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 1,
        Tags.MCX_SEED: RANDOM_SEED
    })

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
        Tags.KWAVE_PROPERTY_INITIAL_PRESSURE_SMOOTHING: False,
    })

    settings.set_reconstruction_settings({
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1480,
        Tags.SPACING_MM: 0.1
    })

    device = sp.InVision256TF(device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2,
                                                             VOLUME_PLANAR_DIM_IN_MM/2,
                                                             VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2]),
                              field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, -15, 15]
                           )
                           )
    SIMUATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMUATION_PIPELINE, settings, device)

    p0 = sp.load_data_field(path_manager.get_hdf5_file_save_path() + settings[Tags.VOLUME_NAME] + ".hdf5",
                                 sp.Tags.DATA_FIELD_INITIAL_PRESSURE, 800)
    recon = sp.load_data_field(path_manager.get_hdf5_file_save_path() + settings[Tags.VOLUME_NAME] + ".hdf5",
                                 sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA, 800)

    np.savez(f"sims/{name}.npz",
             p0=p0,
             recon=recon,
             mua=mua,
             mus=mus)

if __name__ == "__main__":

    data_files = glob.glob("data/*.npz")
    for data_file in data_files:
        name = data_file.split("/")[-1].split("\\")[-1][:-4]
        print(data_file)
        data = np.load(data_file)
        wavelengths = data["wavelengths"]
        mua = data["mua"]
        mua_std = data["mua_std"]
        mus_fit = data["mus_fit"]

        wl_idx = np.argwhere(wavelengths == WAVELENGTH)

        simulate_phantom(f"{name}_low", mua[wl_idx] - mua_std[wl_idx], mus_fit[wl_idx])
        simulate_phantom(f"{name}_mean", mua[wl_idx], mus_fit[wl_idx])
        simulate_phantom(f"{name}_high", mua[wl_idx] + mua_std[wl_idx], mus_fit[wl_idx])
