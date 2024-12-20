from .plan import Plan, Beam, DoseGrid
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import dose_kernels


class IMRTControlPoint:

    def __init__(self, mu, mlc, ga, ca, ta, xjaws, yjaws):
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


    def DensityFromHU(self):
                
        density_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "HU_RLSP.csv"))
        df_density = pd.read_csv(density_table_path)

        hu_curve = df_density["HU"].to_numpy()
        density_curve = df_density["RLSP"].to_numpy()
        
        density = np.array(np.interp(self.HU, hu_curve, density_curve), dtype=np.single)
        
        return density


    def computeIMRTPlan(self, plan, gpu_id=0):
            
        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.Density = self.DensityFromHU()

        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")

        for beam in plan.beam_list:
            beam_dose = np.zeros(self.Density.shape, dtype=np.single)

            for cp in beam.cp_list:
                cp_dose = dose_kernels.photon_dose_cuda(
                    np.array(self.Density, dtype=np.single), 
                    np.array(beam.iso - self.origin, dtype=np.single), 
                    cp.mu, 
                    np.array(cp.mlc, dtype=np.single),
                    cp.ga, 
                    cp.ca, 
                    cp.ta, 
                    0.0,
                    0.0, 
                    self.spacing[0],
                    gpu_id)
                beam_dose += cp_dose * cp.mu

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

    def __init__(self):
        
        super().__init__()

        mlc_geometry_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "mlc_geometry.csv"))
        mlc_geometry = pd.read_csv(mlc_geometry_path)

        self.mlc_index = mlc_geometry["mlc_pair_index"].to_numpy()
        self.mlc_widths = mlc_geometry["width"].to_numpy()
        self.mlc_offsets = mlc_geometry["center_offset"].to_numpy()
        self.mlc_thickness = mlc_geometry["thickness"].to_numpy()
        self.mlc_distance_to_source = mlc_geometry["distance_to_source"].to_numpy()
        self.n_mlc_pairs = len(self.mlc_index)


    def transmissionArrays(self, cp):

        xjaws_mask, yjaws_mask, mlc_mask = None, None, None
        if cp.xjaws is not None:
            points = [(cp.xjaws[0], -250), (cp.xjaws[0], 250), (cp.xjaws[1], 250), (cp.xjaws[1], -250)]
            xjaws_mask, x_coords, y_coords = polygon_to_mask(points)

        if cp.yjaws is not None:
            points = [(-250, cp.yjaws[0]), (250, cp.yjaws[0]), (250, cp.yjaws[1]), (-250, cp.yjaws[1])]
            yjaws_mask, x_coords, y_coords = polygon_to_mask(points)

        if cp.mlc is not None:
            points = []
            for i in range(0, len(cp.mlc)):
                points.append((cp.mlc[i], self.beam_model.mlc_offsets[i] - self.beam_model.mlc_widths[i] / 2))
                points.append((cp.mlc[i], self.beam_model.mlc_offsets[i] + self.beam_model.mlc_widths[i] / 2))
            mlc_mask, x_coords, y_coords = polygon_to_mask(points)

        return xjaws_mask, yjaws_mask, mlc_mask


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

                        mlc = np.reshape(mlc, (2, self.n_mlc_pairs))
                        mlc = np.array(np.vstack((mlc, self.mlc_offsets.reshape(1, -1), self.mlc_widths.reshape(1, -1))), dtype=np.single)
                        mlc = np.transpose(mlc)

                        control_point = IMRTControlPoint(mu * total_mu, mlc, ga, ca, ta, xjaws, yjaws)
                        
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
