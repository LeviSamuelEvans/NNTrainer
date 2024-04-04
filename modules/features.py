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
    representation : bool
        A boolean indicating whether to use multiple four-vector representations in the model.
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
                if not feature_config["feature_maker"]:
                    logging.error(
                        "Feature maker not specified in the configuration file! \n\
                            Please specify the feature maker in the configuration file. \n\
                            Exiting..."
                    )
                    return

                feature_maker = FeatureFactory.make(
                    max_particles=feature_config["max_particles"],
                    n_leptons=feature_config["n_leptons"],
                    extra_feats=feature_config.get("extra_feats", None),
                    representation=feature_config.get("use_representations", False),
                )
                if feature_maker.extra_feats and config_dict["preparation"]["use_extra_feats"]:
                    logging.info(f"Feature maker successfully created with the following extra features: {feature_maker.extra_feats}")
                if feature_maker.representation and config_dict["preparation"]["use_representations"]:
                    logging.info("Using multiple four-vector representations in the model.")
                signal_fvectors = feature_maker.get_four_vectors(signal_data)
                background_fvectors = feature_maker.get_four_vectors(background_data)
                logging.info("Four-vectors successfully constructed!")

        return signal_fvectors, background_fvectors

    @staticmethod
    def make(max_particles, n_leptons, extra_feats=None, representation=False):
        """Create an instance of the FeatureMaker class.

        Parameters
        ----------
        max_particles : int
            The maximum number of particles.
        n_leptons : int
            The number of leptons.
        extra_feats : list, optional
            A list of extra features to include. Defaults to None.
        representation : bool, optional
            A boolean indicating whether to use multiple four-vector
            representations in the model. Defaults to False.

        Returns
        -------
        FeatureMaker
            An instance of the FeatureMaker class.
        """
        return FeatureMaker(max_particles, n_leptons, extra_feats, representation)


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

    def __init__(self, max_particles, n_leptons, extra_feats=None, representation=False):
        self.max_particles = max_particles
        self.n_leptons = n_leptons
        self.extra_feats = extra_feats if extra_feats else []
        self.representation = representation

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
            # TODO: move away from using eval here...
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
        # lepton and jet info
        lep_pt = self.compute_lepton_arrays(sample, "pt")
        lep_eta = self.compute_lepton_arrays(sample, "eta")
        lep_phi = self.compute_lepton_arrays(sample, "phi")
        lep_e = self.compute_lepton_arrays(sample, "e")

        max_jets = self.max_particles - self.n_leptons
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

        p_extra = {}

        lep_btag = np.zeros_like(lep_e)

        if self.extra_feats is not None:
            feat = self.extra_feats
            if feat == "btag":
                jet_btag = self.get_jet_features(sample, "jet_tagWeightBin_DL1r_Continuous", max_jets)
                p_extra["btag"] = np.concatenate((lep_btag, jet_btag), axis=-1)
            # elif feat == ["jet_charge"]:
            #     p_extra["jet_charge"] = self.get_jet_features(sample, "jet_charge", max_jets)
            #     p_extra["jet_charge"] = np.hstack((np.zeros((len(lep_pt), 1)), p_extra["jet_charge"]), axis=-1)
            else:
                logging.error(f"Feature {feat} not recognized! Skipping...")

        four_vectors = None
        # add extra features to the four-vectors
        if self.extra_feats:
            if self.representation:
                four_vectors= np.stack((p_px, p_py, p_pz, p_e, p_pt, p_eta, p_phi), axis=-1)
                p_extra[feat] = np.expand_dims(p_extra[feat], axis=-1)
                four_vectors = np.concatenate((four_vectors, p_extra[feat]), axis=-1)
            else:
                p_extra[feat] = np.expand_dims(p_extra[feat], axis=-1)
                four_vectors = np.stack((p_px, p_py, p_pz, p_e), axis=-1)
                four_vectors = np.concatenate((four_vectors, p_extra[feat]), axis=-1)
                # TODO: validate with degbugging logs
        else:
            logging.info("Using only the four-vectors in the model.")
            four_vectors = np.stack((p_px, p_py, p_pz, p_e), axis=-1)

        # Return the array of four vectors and extra features
        return four_vectors

    def get_matched_objetcs():
        return None  # TODO: implement method (for the Higgs decay angles and other matching tasks)
        # once we have the matched objects, boost to relevant rest frame
        # and calculate decay angles of the Higgs, top,W's, and gluons
        # then save the decay angles as extra features and concat to the four-vectors
        # could also add the jet substructure variables and energy flow correlations after
        # the decay angles, alongside some invariant mass variables?
        # add met to extra variables as well, with flag for missing met
