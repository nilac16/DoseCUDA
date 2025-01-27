from .plan import Plan, Beam, DoseGrid
import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pydicom as pyd
import pkg_resources
import dose_kernels
import os
import SimpleITK as sitk

      
class IMPTDoseGrid(DoseGrid):

    def __init__(self):
        super().__init__()
        self.RLSP = []

    def RLSPFromHU(self, machine_name):

        rlsp_table_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", machine_name, "HU_RLSP.csv"))
        df_rlsp = pd.read_csv(rlsp_table_path)

        hu_curve = df_rlsp["HU"].to_numpy()
        rlsp_curve = df_rlsp["RLSP"].to_numpy()
        
        rlsp = np.array(np.interp(self.HU, hu_curve, rlsp_curve), dtype=np.single)
        
        return rlsp
    
    def computeIMPTPlan(self, plan, gpu_id=0):

        self.beam_doses = []
        self.dose = np.zeros(self.size, dtype=np.single)
        self.RLSP = self.RLSPFromHU(plan.machine_name)

        #check if spacing is isotropic
        if self.spacing[0] != self.spacing[1] or self.spacing[0] != self.spacing[2]:
            raise Exception("Spacing must be isotropic for IMPT dose calculation - consider resampling CT")

        for beam in plan.beam_list:

            beam_wet = dose_kernels.proton_raytrace_cuda(np.array(self.RLSP, dtype=np.single), 
                                                         np.array(beam.iso - self.origin, dtype=np.single), 
                                                         beam.gantry_angle, 
                                                         beam.couch_angle, 
                                                         self.spacing[0], 
                                                         gpu_id)

            beam_dose = dose_kernels.proton_spot_cuda(np.array(beam_wet, dtype=np.single),
                                                    np.array(self.RLSP, dtype=np.single),
                                                    np.array(beam.iso - self.origin, dtype=np.single), 
                                                    beam.gantry_angle, 
                                                    beam.couch_angle, 
                                                    np.array(beam.spot_list, dtype=np.single),
                                                    self.spacing[0], 
                                                    np.array(plan.lut_depths, dtype=np.single), 
                                                    np.array(plan.lut_sigmas,dtype=np.single),
                                                    np.array(plan.lut_idds, dtype=np.single),
                                                    np.array(plan.divergence_params, dtype=np.single),
                                                    gpu_id)

            self.beam_doses.append(beam_dose * plan.n_fractions)
            self.dose += beam_dose

        self.dose *= plan.n_fractions

    def writeWETNRRD(self, plan, wet_path, gpu_id=0):

        if not wet_path.endswith(".nrrd"):
            raise Exception("WET path must have .nrrd extension")

        self.RLSP = self.RLSPFromHU()
        fw = sitk.ImageFileWriter()

        for i, beam in enumerate(plan.beam_list):

            beam_wet = dose_kernels.proton_raytrace_cuda(self.RLSP, 
                                                         np.array(beam.iso - self.origin, dtype=np.single), 
                                                         beam.gantry_angle, 
                                                         beam.couch_angle, 
                                                         self.spacing[0], 
                                                         gpu_id)

            HU_img = sitk.GetImageFromArray(beam_wet)
            HU_img.SetOrigin(self.origin)
            HU_img.SetSpacing(self.spacing)

            fw.SetFileName(wet_path.replace(".nrrd", "_beam%02i.nrrd" % (i+1)))
            fw.Execute(HU_img)


class IMPTBeam(Beam):

    def __init__(self):
        super().__init__()

        self.spot_list = []
        self.n_spots = 0

    def addSpotData(self, cp, energy_id):

        spm = np.reshape(np.array(cp.ScanSpotPositionMap), (-1, 2))
        mus = np.array(cp.ScanSpotMetersetWeights)
        energy_id_array = np.full(mus.size, energy_id)

        spot_list = np.array(np.column_stack((spm, mus, energy_id_array)), dtype=np.single)
        self.spot_list = np.vstack((self.spot_list, spot_list)) if self.n_spots > 0 else spot_list
        self.n_spots += mus.size

    def changeSpotEnergy(self, energy_id):
        self.spot_list[:, 3] = energy_id

    def addSingleSpot(self, x, y, mu, energy_id):
        spot = np.array([x, y, mu, energy_id], dtype=np.single)
        self.spot_list = np.vstack((self.spot_list, spot)) if self.n_spots > 0 else spot
        self.n_spots += 1


class IMPTPlan(Plan):

    def __init__(self, machine_name = "HitachiProbeatJHU"):
        super().__init__()
        self.machine_name = machine_name
        self.lut_depths = []
        self.lut_sigmas = []
        self.lut_idds = []
        self.divergence_params = []
        energy_list_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", machine_name, "energies.csv"))
        self.energy_table = pd.read_csv(energy_list_path)
        self.energy_labels = []
        self.loadLUT()

    def loadLUT(self):

        energy_list_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", self.machine_name, "energies.csv"))
        self.energy_table = pd.read_csv(energy_list_path)

        self.energy_labels = self.energy_table["energy_label"].to_numpy()
        energy_ids = self.energy_table["index"].to_numpy()

        self.divergence_params = []
        self.lut_depths = []
        self.lut_sigmas = []
        self.lut_idds = []

        for energy_label, energy_id in zip(self.energy_labels, energy_ids):

            lut_depths = []
            lut_sigmas = []
            lut_idds = []
            divergence_params = []
            
            lut_path = pkg_resources.resource_filename(__name__, os.path.join("lookuptables", "protons", self.machine_name, "energy_%03d.csv" % energy_id))

            with open(lut_path, "r") as f:
                f.readline() # header
                line = f.readline()
                parts = line.split(",")
                divergence_params = [float(part) for part in parts]
                f.readline() # blank
                f.readline() # header
                for line in f:
                    parts = line.split(",")
                    lut_depths.append(float(parts[0]))
                    lut_sigmas.append(float(parts[1]))
                    lut_idds.append(float(parts[2]))
            
            self.divergence_params.append(divergence_params)    
            self.lut_depths.append(lut_depths)
            self.lut_sigmas.append(lut_sigmas)
            self.lut_idds.append(lut_idds)

        self.divergence_params = np.array(self.divergence_params, dtype=np.single)
        self.lut_depths = np.array(self.lut_depths, dtype=np.single)
        self.lut_sigmas = np.array(self.lut_sigmas, dtype=np.single)
        self.lut_idds = np.array(self.lut_idds, dtype=np.single)

        self.lut_depths = self.lut_depths[:, 0:400]
        self.lut_sigmas = self.lut_sigmas[:, 0:400]
        self.lut_idds = self.lut_idds[:, 0:400]

    def energyIDFromLabel(self, energy_label):

        energy_row = self.energy_table[self.energy_table["energy_label"] == energy_label]

        if len(energy_row) != 1:
            raise Exception("Energy ID not found for %.2f" % energy_label)

        return energy_row.index[0]

    def readPlanDicom(self, plan_path):

        ds = pyd.dcmread(plan_path)
        n_beams = len(ds.IonBeamSequence)
        self.n_fractions = float(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        self.beam_list = []
        self.n_beams = 0

        for i in range(n_beams):

            ibs = ds.IonBeamSequence[i]
            
            beam = IMPTBeam()
            
            beam.gantry_angle = float(ibs.IonControlPointSequence[0].GantryAngle)
            beam.couch_angle = float(ibs.IonControlPointSequence[0].PatientSupportAngle)
            beam.iso = np.array(ibs.IonControlPointSequence[0].IsocenterPosition, dtype=np.single)
            
            for j in range(len(ibs.IonControlPointSequence)):
            
                if j % 2 == 0:
            
                    cp = ibs.IonControlPointSequence[j]

                    energy_id = self.energyIDFromLabel(float(cp.NominalBeamEnergy))

                    beam.addSpotData(cp, energy_id)

            self.addBeam(beam)