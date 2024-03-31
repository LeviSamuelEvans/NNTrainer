import numpy as np
import pandas as pd
import ast
import logging

"""
TODO:
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
    """
    A class that provides methods for extracting features from input samples.

    Methods
    -------
    extract_features(config_dict, signal_data, background_data)
        Extract features from signal and background data based on the configuration.
    make(max_particles, n_leptons, extra_feats=None)
        Create an instance of the FeatureMaker class.

    Attributes
    ----------
    max_particles : int
        The maximum number of particles.
    n_leptons : int
        The number of leptons.
    extra_feats : list
        A list of extra features to include.
    """
    @staticmethod
    def extract_features(config_dict, signal_data, background_data):
        """Extract features from signal and background data based on the configuration.

        Parameters
        ----------
        config_dict : dict
            The configuration dictionary containing feature extraction settings.
        signal_data : numpy.ndarray
            The signal data.
        background_data : numpy.ndarray
            The background data.

        Returns
        -------
        tuple
            A tuple containing the extracted signal and background four-vectors.
                signal_fvectors : numpy.ndarray
                    The extracted signal four-vectors.
                background_fvectors : numpy.ndarray
                    The extracted background four-vectors.
        """
        signal_fvectors = None
        background_fvectors = None

        if config_dict["preparation"]["feature_maker"]:
            feature_config = config_dict["preparation"]
            if feature_config["feature_type"] == "4-vectors":
                logging.info(
                    "Proceeding to construct the four-vectors of the objects.."
                )
                feature_maker = FeatureFactory.make(
                    max_particles=feature_config["max_particles"],
                    n_leptons=feature_config["n_leptons"],
                    extra_feats=feature_config.get("extra_feats"),
                )
                signal_fvectors = feature_maker.get_four_vectors(signal_data)
                background_fvectors = feature_maker.get_four_vectors(background_data)
                logging.info("Four-vectors successfully constructed!")

        return signal_fvectors, background_fvectors

    @staticmethod
    def make(max_particles, n_leptons, extra_feats=None):
        """Create an instance of the FeatureMaker class.

        Parameters
        ----------
        max_particles : int
            The maximum number of particles.
        n_leptons : int
            The number of leptons.
        extra_feats : list, optional
            A list of extra features to include. Defaults to None.

        Returns
        -------
        FeatureMaker
            An instance of the FeatureMaker class.
        """
        return FeatureMaker(max_particles, n_leptons, extra_feats)


class FeatureMaker:
    """
    A class that provides methods for computing features from input samples.

    Attributes
    ----------
    max_particles : int
        The maximum number of particles.
    n_leptons : int
        The number of leptons.
    extra_feats : list, optional
        A list of extra features to include. Defaults to None.
    """

    def __init__(self, max_particles, n_leptons, extra_feats=None):
        self.max_particles = max_particles
        self.n_leptons = n_leptons
        self.extra_feats = extra_feats if extra_feats else []

    def get_jet_features(self, sample, feature_name, max_jets):
        """Get jet features from the input sample.

        Parameters
        ----------
        sample : numpy.ndarray
            The input sample.
        feature_name : str
            The name of the jet feature.
        max_jets : int
            The maximum number of jets.

        Returns
        -------
        numpy.ndarray
            The jet features.
        """
        # TODO: move to multi-dim arrays
        feature_columns = [f"{feature_name}_{i+1}" for i in range(max_jets)]
        jet_features = sample[feature_columns].values
        return jet_features

    def compute_lepton_arrays(self, sample, array_name):
        """Compute lepton arrays from the input sample.

        Parameters
        ----------
        sample : numpy.ndarray
            The input sample.
        array_name : str
            The name of the array to compute.

        Returns
        -------
        numpy.ndarray
            The computed lepton arrays.
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
            # convert the string representation of the arrays to actual arrays
            el_data = ast.literal_eval(el_data)
            mu_data = ast.literal_eval(mu_data)

            # get the total number of leptons
            n_leptons = n_electrons + n_muons

            # make sure we have the right number of leptons
            assert n_leptons == self.n_leptons
            out[i, :n_electrons] = el_data[:n_electrons]
            out[i, n_electrons:n_leptons] = mu_data[:n_muons]
        return out

    def get_four_vectors(self, sample):
        """Compute four-vectors and extra features from the input sample.

        Parameters
        ----------
        sample : numpy.ndarray
            The input sample.

        Returns
        -------
        numpy.ndarray
            The computed four-vectors and extra features.
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
        return None  # TODO: implement method (for the Higgs decay angles and other matching tasks)
