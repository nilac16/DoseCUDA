from .plan import Plan, Beam, DoseGrid, VolumeObject
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import dose_kernels


class IMRTBeamModel:

    def __init__(self, path_to_model):
        
        mlc_geometry_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "mlc_geometry.csv"))
        mlc_geometry = pd.read_csv(mlc_geometry_path)

        self.mlc_index = mlc_geometry["mlc_pair_index"].to_numpy()
        self.mlc_widths = mlc_geometry["width"].to_numpy()
        self.mlc_offsets = mlc_geometry["center_offset"].to_numpy()
        self.n_mlc_pairs = len(self.mlc_index)

        kernel_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "kernel.csv"))
        kernel = pd.read_csv(kernel_path)
        self.kernel = np.array(kernel.to_numpy(), dtype=np.single)

        machine_parameters_path = pkg_resources.resource_filename(__name__, os.path.join(path_to_model, "machine_parameters.csv"))

        self.output_factor_equivalent_squares = None
        self.output_factor_values = None
        self.mu_calibration = None
        self.primary_source_distance = None
        self.scatter_source_distance = None
        self.mlc_distance = None
        self.scatter_source_weight = None
        self.electron_attenuation = None
        self.primary_source_size = None
        self.scatter_source_size = None
        self.profile_radius = None
        self.profile_intensities = None
        self.profile_softening = None
        self.spectrum_attenuation_coefficients = None
        self.spectrum_primary_weights = None
        self.spectrum_scatter_weights = None
        self.electron_source_weight = None
        self.has_xjaws = None
        self.has_yjaws = None
        self.electron_fitted_dmax = None
        self.jaw_transmission = None
        self.mlc_transmission = None

        for line in open(machine_parameters_path):

            if line.startswith('output_factor_equivalent_squares'):
                self.output_factor_equivalent_squares = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('output_factor_values'):
                self.output_factor_values = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('mu_calibration'):
                self.mu_calibration = float(line.split(',')[1])

            if line.startswith('primary_source_distance'):
                self.primary_source_distance = float(line.split(',')[1])

            if line.startswith('scatter_source_distance'):
                self.scatter_source_distance = float(line.split(',')[1])

            if line.startswith('mlc_distance'):
                self.mlc_distance = float(line.split(',')[1])

            if line.startswith('scatter_source_weight'):
                self.scatter_source_weight = float(line.split(',')[1])

            if line.startswith('electron_attenuation'):
                self.electron_attenuation = float(line.split(',')[1])

            if line.startswith('primary_source_size'):
                self.primary_source_size = float(line.split(',')[1])

            if line.startswith('scatter_source_size'):
                self.scatter_source_size = float(line.split(',')[1])

            if line.startswith('profile_radius'):
                self.profile_radius = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('profile_intensities'):
                self.profile_intensities = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('profile_softening'):
                self.profile_softening = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('spectrum_attenuation_coefficients'):
                self.spectrum_attenuation_coefficients = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('spectrum_primary_weights'):
                self.spectrum_primary_weights = np.array(line.split(',')[1:], dtype=np.single)

            if line.startswith('spectrum_scatter_weights'):
                self.spectrum_scatter_weights = np.array(line.split(',')[1:], dtype=np.single)
            
            if line.startswith('electron_source_weight'):
                self.electron_source_weight = float(line.split(',')[1])

            if line.startswith('has_xjaws'):
                self.has_xjaws = bool(line.split(',')[1])

            if line.startswith('has_yjaws'):
                self.has_yjaws = bool(line.split(',')[1])

            if line.startswith('electron_fitted_dmax'):
                self.electron_fitted_dmax = float(line.split(',')[1])

            if line.startswith('jaw_transmission'):
                self.jaw_transmission = float(line.split(',')[1])

            if line.startswith('mlc_transmission'):
                self.mlc_transmission = float(line.split(',')[1])


        if self.output_factor_equivalent_squares is None:
            raise Exception("output_factor_equivalent_squares not found in machine_parameters.csv")
        
        if self.output_factor_values is None:
            raise Exception("output_factor_values not found in machine_parameters.csv")
        
        if self.mu_calibration is None:
            raise Exception("mu_calibration not found in machine_parameters.csv")

        if self.primary_source_distance is None:
            raise Exception("primary_source_distance not found in machine_parameters.csv")
        
        if self.scatter_source_distance is None:
            raise Exception("scatter_source_distance not found in machine_parameters.csv")
        
        if self.mlc_distance is None:
            raise Exception("mlc_distance not found in machine_parameters.csv")
        
        if self.scatter_source_weight is None:
            raise Exception("scatter_source_weight not found in machine_parameters.csv")
        
        if self.electron_attenuation is None:
            raise Exception("electron_mass_attenuation_coefficient not found in machine_parameters.csv")
        
        if self.primary_source_size is None:
            raise Exception("primary_source_size not found in machine_parameters.csv")
        
        if self.scatter_source_size is None:
            raise Exception("scatter_source_size not found in machine_parameters.csv")
        
        if self.profile_radius is None:
            raise Exception("profile_radius not found in machine_parameters.csv")
        
        if self.profile_intensities is None:
            raise Exception("profile_intensities not found in machine_parameters.csv")
        
        if self.profile_softening is None:
            raise Exception("profile_softening not found in machine_parameters.csv")
        
        if self.spectrum_attenuation_coefficients is None:
            raise Exception("spectrum_attenuation_coefficients not found in machine_parameters.csv")
        
        if self.spectrum_primary_weights is None:
            raise Exception("spectrum_primary_weights not found in machine_parameters.csv")
        
        if self.spectrum_scatter_weights is None:
            raise Exception("spectrum_scatter_weights not found in machine_parameters.csv")
        
        if self.electron_source_weight is None:
            raise Exception("electron_source_weight not found in machine_parameters.csv")
        
        if self.has_xjaws is None:
            raise Exception("has_xjaws not found in machine_parameters.csv")
        
        if self.has_yjaws is None:
            raise Exception("has_yjaws not found in machine_parameters.csv")
        
        if self.electron_fitted_dmax is None:
            raise Exception("electron_fitted_dmax not found in machine_parameters.csv")
        
        if self.jaw_transmission is None:
            raise Exception("jaw_transmission not found in machine_parameters.csv")
        
        if self.mlc_transmission is None:
            raise Exception("mlc_transmission not found in machine_parameters.csv")
        
    def outputFactor(self, cp):
    
        min_y = 10000.0
        max_y = -10000.0
        max_x_diff = 0.0
        area = 0.0
        
        for i in range(cp.mlc.shape[0]):

            x1 = cp.mlc[i, 0]
            x2 = cp.mlc[i, 1]
            y_offset = cp.mlc[i, 2]
            y_width = cp.mlc[i, 3]

            area += (x2 - x1) * y_width

            if (x2 - x1) > 3.0 and (x2 - x1) > max_x_diff:
                max_x_diff = x2 - x1

            if (x2 - x1) > 3.0:
                if (y_offset - y_width / 2.0) < min_y:
                    min_y = (y_offset - y_width / 2.0)
                if (y_offset + y_width / 2.0) > max_y:
                    max_y = (y_offset + y_width / 2.0)

        perimeter = 2 * (max_y - min_y) + 2 * max_x_diff

        equivalent_square = 4 * area / perimeter

        output_factor = np.interp(equivalent_square, self.output_factor_equivalent_squares, self.output_factor_values)

        return output_factor
       

class IMRTControlPoint:

    def __init__(self, iso, mu, mlc, ga, ca, ta, xjaws, yjaws):
        self.iso = iso
        self.mu = mu
        self.mlc = mlc
        self.ga = ga
        self.ca = ca
        self.ta = ta
        self.xjaws = xjaws
        self.yjaws = yjaws


class IMRTDoseGrid(DoseGrid):
        
    def __init__(self):
        super().__init__()
        self.Density = []

    def DensityFromHU(self, machine_name):
                
        density_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "photons", machine_name, "HU_Density.csv"))
        df_density = pd.read_csv(density_table_path)

        hu_curve = df_density["HU"].to_numpy()
        density_curve = df_density["Density"].to_numpy()
        
        density = np.array(np.interp(self.HU, hu_curve, density_curve), dtype=np.single)
        
        return density

    def computeIMRTPlan(self, plan, gpu_id=0):
            
        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.Density = self.DensityFromHU(plan.machine_name)

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")
        
        density_object = VolumeObject()
        density_object.voxel_data = np.array(self.Density, dtype=np.single)
        density_object.origin = np.array(self.origin, dtype=np.single)
        density_object.spacing = np.array(self.spacing, dtype=np.single)

        for beam in plan.beam_list:
            beam_dose = np.zeros(self.Density.shape, dtype=np.single)
            beam_model = plan.beam_model
            for cp in beam.cp_list:
                output_factor = plan.beam_model.outputFactor(cp)
                cp_dose = dose_kernels.photon_dose_cuda(beam_model, density_object, cp, gpu_id)
                beam_dose += cp_dose * output_factor
                
            self.beam_doses.append(beam_dose)
            self.dose += beam_dose

        self.dose *= plan.n_fractions


class IMRTBeam(Beam):

    def __init__(self):
        super().__init__()
        self.cp_list = []
        self.n_cps = 0

    def addControlPoint(self, cp):
        self.cp_list.append(cp)
        self.n_cps += 1


class IMRTPlan(Plan):

    def __init__(self, machine_name = "VarianTrueBeamHF"):
        
        super().__init__()

        self.machine_name = machine_name
        self.beam_model = IMRTBeamModel(os.path.join("lookuptables", "photons", machine_name))

    def readPlanDicom(self, plan_path):

        ds = pyd.dcmread(plan_path)
        total_mu = ds.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamMeterset

        self.n_fractions = float(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        self.beam_list = []
        self.n_beams = 0

        for beam in ds.BeamSequence:
            if beam.TreatmentDeliveryType == "TREATMENT":

                imrt_beam = IMRTBeam()
                
                for cp in beam.ControlPointSequence:
                    cpi = cp.ControlPointIndex
                    if cpi == 0:
                        
                        imrt_beam.iso = np.array([cp.IsocenterPosition[0], cp.IsocenterPosition[1], cp.IsocenterPosition[2]], dtype=np.single)
                        cumulative_mu = 0.0
                        ca = cp.BeamLimitingDeviceAngle
                        ta = cp.PatientSupportAngle
                        ga = cp.GantryAngle

                        xjaws, yjaws, mlc = None, None, None
                        for ps in cp.BeamLimitingDevicePositionSequence:
                            if ps.RTBeamLimitingDeviceType == 'X':
                                xjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                            if ps.RTBeamLimitingDeviceType == 'Y':
                                yjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                            if ps.RTBeamLimitingDeviceType == 'MLCX':
                                mlc = np.array(cp.BeamLimitingDevicePositionSequence[2].LeafJawPositions, dtype=np.single)

                    else:

                        mu = cp.CumulativeMetersetWeight - cumulative_mu

                        if mlc is not None:
                            mlc = np.reshape(mlc, (2, self.beam_model.n_mlc_pairs))
                            mlc = np.array(np.vstack((mlc, self.beam_model.mlc_offsets.reshape(1, -1), self.beam_model.mlc_widths.reshape(1, -1))), dtype=np.single)
                            mlc = np.transpose(mlc)

                        else:

                            mlc = np.zeros((2, self.beam_model.n_mlc_pairs), dtype=np.single)

                            if xjaws is not None:
                                mlc[0, :] = xjaws[0]
                                mlc[1, :] = xjaws[1]

                            if yjaws is not None:
                                for i in range(self.beam_model.n_mlc_pairs):
                                    if(self.beam_model.mlc_offsets[i] < yjaws[0]) or (self.beam_model.mlc_offsets[i] > yjaws[1]):
                                        mlc[0, i] = 0.0
                                        mlc[1, i] = 0.0
                            
                            mlc = np.array(np.vstack((mlc, self.beam_model.mlc_offsets.reshape(1, -1), self.beam_model.mlc_widths.reshape(1, -1))), dtype=np.single)
                            mlc = np.transpose(mlc)

                        control_point = IMRTControlPoint(imrt_beam.iso, mu * total_mu, mlc, ga, ca, ta, xjaws, yjaws)
                        
                        imrt_beam.addControlPoint(control_point)

                        if hasattr(cp, 'GantryAngle'):
                            ga = cp.GantryAngle

                        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                            xjaws, yjaws, mlc = None, None, None
                            for ps in cp.BeamLimitingDevicePositionSequence:
                                if ps.RTBeamLimitingDeviceType == 'X':
                                    xjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                                if ps.RTBeamLimitingDeviceType == 'Y':
                                    yjaws = np.array(cp.BeamLimitingDevicePositionSequence[1].LeafJawPositions, dtype=np.single)
                                if ps.RTBeamLimitingDeviceType == 'MLCX':
                                    mlc = np.array(cp.BeamLimitingDevicePositionSequence[2].LeafJawPositions, dtype=np.single)

                        cumulative_mu = cp.CumulativeMetersetWeight
                
                self.addBeam(imrt_beam)

    def addSquareField(self, dimx=10, dimy=10, mu=100, gantry_angle=0.0, collimator_angle=0.0, table_angle=0.0):

        imrt_beam = IMRTBeam()
        iso = np.array([0.0, 0.0, 0.0], dtype=np.single)

        dimx *= 10.0 # convert to mm
        dimy *= 10.0 # convert to mm

        # create a square field
        mlc = np.zeros((2, self.beam_model.n_mlc_pairs), dtype=np.single)
        mlc[0, :] = -dimx / 2.0
        mlc[1, :] = dimx / 2.0

        for i in range(self.beam_model.n_mlc_pairs):
            if(self.beam_model.mlc_offsets[i] < -(dimy / 2.0)) or (self.beam_model.mlc_offsets[i] > (dimy / 2.0)):
                mlc[0, i] = 0.0
                mlc[1, i] = 0.0

        mlc = np.array(np.vstack((mlc, self.beam_model.mlc_offsets.reshape(1, -1), self.beam_model.mlc_widths.reshape(1, -1))), dtype=np.single)
        mlc = np.transpose(mlc)

        jawx = np.array([-dimx / 2.0, dimx / 2.0], dtype=np.single)
        jawy = np.array([-dimy / 2.0, dimy / 2.0], dtype=np.single)


        cp = IMRTControlPoint(iso, mu, mlc, gantry_angle, collimator_angle, table_angle, jawx, jawy)
        imrt_beam.addControlPoint(cp)
        self.addBeam(imrt_beam)
