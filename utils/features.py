# Future script to perform feature engineering on the original data
import numpy as np
import pandas as pd

"""
TODO:
    - particle 4-vectors (jets,leptons) (legacy transformer)
    - Higgs decay angles (use bitwise operations of truth matching) (old CMS BDT for ttH)
        - boost to rest frame of Higgs
        - boost to rest frame of top (and take left-over)
    - Jet substructure variables
    - Energy flow correlations (no clue: https://arxiv.org/pdf/1305.0007.pdf; 
      https://jduarte.physics.ucsd.edu/iaifi-summer-school/1.1_tabular_data_efps.html)
    - 2neutrino scanning method (https://indico.cern.ch/event/1032082/#2-dilepton-ttbar-reconstruction)
    - Extra Fox-Wolfram moments (https://arxiv.org/pdf/1212.4436.pdf)
"""

# Objects for matching
matching_objects = ["H", "b_had_top", "b_lep_top", "W"]


class FeatureFactory:
    @staticmethod
    def make(max_particles, n_leptons, extra_feats=None):
        return FeatureMaker(max_particles, n_leptons, extra_feats)


class FeatureMaker:
    """
    A class that provides methods for computing features from input samples.

    Args:
        max_particles (int): The maximum number of particles.
        n_leptons (int): The number of leptons.
        extra_feats (list, optional): A list of extra features to include. Defaults to None.

    Methods:
        compute_lepton_arrays(sample, array_name):
            Computes the lepton arrays from the input sample.

        get_four_vectors(sample):
            Computes the four vectors and extra features from the input sample.

    """

    def __init__(self, max_particles, n_leptons, extra_feats=None):
        self.max_particles = max_particles
        self.n_leptons = n_leptons
        self.extra_feats = extra_feats if extra_feats else []

    def get_jet_features(self, sample, feature_name, max_jets):
        # get all the jet features from our input files
        ## FUTURE -> move to mutli-dim h5 file as will be easier and neater
        feature_columns = [f"{feature_name}_{i+1}" for i in range(max_jets)]
        jet_features = sample[feature_columns].values
        return jet_features

    def compute_lepton_arrays(self, sample, array_name):
        """
        Computes the lepton arrays from the input sample.

        Args:
            sample (numpy.ndarray): The input sample.
            array_name (str): The name of the array to compute.

        Returns:
            numpy.ndarray: The computed lepton arrays.

        """
        # create an empty array of shape (n_samples, n_leptons)
        out = np.full((len(sample), self.n_leptons), np.nan)
        # loop over the samples
        for i, (n_electrons, n_muons, el_data, mu_data) in enumerate(
            zip(
                sample["nElectrons"],
                sample["nMuons"],
                sample["el_" + array_name],
                sample["mu_" + array_name],
            )
        ):
            n_leptons = n_electrons + n_muons
            # make sure we have the right number of leptons
            assert n_leptons == self.n_leptons
            out[i, :n_electrons] = el_data[:n_electrons]
            out[i, n_electrons:n_leptons] = mu_data[:n_muons]
        return out

    def get_four_vectors(self, sample):
        """
        Computes the four vectors and extra features from the input sample.

        Args:
            sample (numpy.ndarray): The input sample.

        Returns:
            numpy.ndarray: The computed four vectors and extra features.

        """
        # lepton info
        lep_pt = self.compute_lepton_arrays(sample, "pt")
        lep_eta = self.compute_lepton_arrays(sample, "eta")
        lep_phi = self.compute_lepton_arrays(sample, "phi")
        lep_e = self.compute_lepton_arrays(sample, "e")

        # max jets is total particles minus the number of leptons
        max_jets = self.max_particles - self.n_leptons

        # jet info ( we already flattened in the HDF5 file so no need here)
        jet_pt = self.get_jet_features(sample, "jet_pt", max_jets)
        jet_eta = self.get_jet_features(sample, "jet_eta", max_jets)
        jet_phi = self.get_jet_features(sample, "jet_phi", max_jets)
        jet_e = self.get_jet_features(sample, "jet_e", max_jets)

        # concat the lepton and jet info
        p_pt = np.hstack((lep_pt, jet_pt))
        p_eta = np.hstack((lep_eta, jet_eta))
        p_phi = np.hstack((lep_phi, jet_phi))
        p_e = np.hstack((lep_e, jet_e))

        # calculate px, py, pz from pt, eta, phi
        p_px = p_pt * np.cos(p_phi)  # cos(phi) = px/pt
        p_py = p_pt * np.sin(p_phi)  # sin(phi) = py/pt
        p_pz = p_pt * np.sinh(p_eta)  # sinh(eta) = pz/pt

        # join together all our four vectors
        four_vectors = np.stack(
            (p_px, p_py, p_pz, p_e), axis=-1
        )  # axis=-1 means stack along the last axis

        # add extra features if specified by user (i.e. the jet tagging weights, etc.)
        if self.extra_feats:
            extra_features = [sample[feat].values for feat in self.extra_feats]
            four_vectors = np.hstack((four_vectors, extra_features))

        # Return the array of four vectors and extra features
        return four_vectors

    def get_matched_objetcs():
        return None  # TODO -> implement this method for the Higgs decay angles
