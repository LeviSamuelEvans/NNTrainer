import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import time
from scipy.spatial import cKDTree

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
        signal_edges = None
        signal_edge_attr = None
        background_edges = None
        background_edge_attr = None

        if config_dict["preparation"]["feature_maker"]:
            feature_config = config_dict["preparation"]
            if feature_config["feature_type"] == "4-vectors":
                logging.info(
                    "FeatureFactory :: Proceeding to construct the four-vectors of the objects.."
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
                    further_representation=feature_config.get(
                        "use_all_representations", False
                    ),
                    include_met=feature_config.get("include_met", False),
                    angular_separation=feature_config.get(
                        "use_angular_separation", False
                    ),
                )

                if (
                    feature_maker.extra_feats
                    and config_dict["preparation"]["use_extra_feats"]
                ):
                    logging.info(
                        f"FeatureFactory :: Feature maker successfully created with the following extra features: {feature_maker.extra_feats}"
                    )
                if (
                    feature_maker.representation
                    and config_dict["preparation"]["use_representations"]
                ):
                    logging.info(
                        "FeatureFactory :: Using multiple four-vector representations in the model."
                    )
                if config_dict["preparation"]["use_all_representations"]:
                    logging.info(
                        "FeatureFactory :: Using all the four-vector representations in the model! :D"
                    )  # add logging of all reps!
                if config_dict["preparation"]["include_met"]:
                    logging.info(
                        "FeatureFactory :: Including the missing transverse energy in the features."
                    )

                signal_fvectors = feature_maker.get_four_vectors(signal_data)
                background_fvectors = feature_maker.get_four_vectors(background_data)
                logging.info("FeatureFactory :: Four-vectors successfully constructed!")

                if config_dict["preparation"]["use_angular_separation"]:
                    logging.info(
                        f"FeatureFactory :: Constructing the angular separation features..."
                    )
                    signal_edges, signal_edge_attr = feature_maker.construct_edges(
                        signal_data
                    )
                    background_edges, background_edge_attr = (
                        feature_maker.construct_edges(background_data)
                    )
                    logging.info(
                        "FeatureFactory :: Angular separation features successfully constructed!"
                    )

        return (
            signal_fvectors,
            background_fvectors,
            signal_edges,
            signal_edge_attr,
            background_edges,
            background_edge_attr,
        )

    @staticmethod
    def make(
        max_particles,
        n_leptons,
        extra_feats=None,
        representation=False,
        further_representation=False,
        include_met=False,
        angular_separation=False,
    ):
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
        further_representation : bool, optional
            A boolean indicating whether to use even more four-vector style features.
        include_met: bool, optional
            A boolean indicating whether to include the missing transverse energy in the features.
        angular_separation : bool, optional
            A boolean indicating whether to use angular separation of the objects.
            Defaults to False.

        Returns
        -------
        FeatureMaker
            An instance of the FeatureMaker class.
        """
        return FeatureMaker(
            max_particles,
            n_leptons,
            extra_feats,
            representation,
            further_representation,
            include_met,
            angular_separation,
        )


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

    def __init__(
        self,
        max_particles,
        n_leptons,
        extra_feats=None,
        representation=False,
        further_representation=False,
        include_met=False,
        angular_separation=False,
    ):
        self.max_particles = max_particles
        self.n_leptons = n_leptons
        self.extra_feats = extra_feats if extra_feats else []
        self.representation = representation
        self.further_representation = further_representation
        self.include_met = include_met
        self.angular_separation = angular_separation
        self.ht_all = None

    def get_event_info(self, sample, info_type):
        """Fetches event-level information.

        Parameters
        ----------
        sample : pandas.DataFrame
            The input data sample.
        info_type : str
            Type of event info to fetch (e.g., 'nJets' or 'HT_all').

        Returns
        -------
        np.ndarray
            The requested event information.
        """
        return sample[info_type].values

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

    # NEW TRIAL
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
        out = np.full((len(sample), self.n_leptons), 0.0)

        # loop over the samples
        for i in range(len(sample)):
            # get the lepton data for the current sample
            lepton_data = []
            for j in range(self.n_leptons):
                if sample[f"el_{array_name}_{j}"].iloc[i] != 0.0:
                    lepton_data.append(sample[f"el_{array_name}_{j}"].iloc[i])
                elif sample[f"mu_{array_name}_{j}"].iloc[i] != 0.0:
                    lepton_data.append(sample[f"mu_{array_name}_{j}"].iloc[i])

            # assign the lepton data to the output array
            out[i, : len(lepton_data)] = lepton_data

        return out

    def _get_basic_four_vectors(self, sample):
        """Get basic four-vectors from the input sample."""
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

        # concatenate the lepton and jet info
        p_pt = np.hstack((lep_pt, jet_pt))
        p_eta = np.hstack((lep_eta, jet_eta))
        p_phi = np.hstack((lep_phi, jet_phi))
        p_e = np.hstack((lep_e, jet_e))

        # calculate px, py, pz from pt, eta, phi
        p_px = p_pt * np.cos(p_phi)
        p_py = p_pt * np.sin(p_phi)
        p_pz = p_pt * np.sinh(p_eta)

        # stack the four vectors
        # basic_four_vectors = np.stack((p_px, p_py, p_pz, p_e), axis=-1)
        return p_px, p_py, p_pz, p_e

    # TODO: refactor this met
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

        epsilon = 1e-8

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

        # calculate additional components for further representations
        p_theta = np.where(p_pt > 0, np.arctan2(p_pt, p_pz), 0)

        # mass with epsilon to avoid invalid values in sqrt
        p_mass = np.where(
            p_pt > 0,
            np.sqrt(np.maximum(p_e**2 - p_px**2 - p_py**2 - p_pz**2, epsilon)),
            0,
        )

        # Calculate E/m and pT/m (NOT USED)
        p_E_over_m = np.where(p_pt > 0, p_e / (p_mass + epsilon), 0)
        p_pT_over_m = np.where(p_pt > 0, p_pt / (p_mass + epsilon), 0)

        # Calculate log(pT) and log(m) (NOT USED)
        p_log_pT = np.where(p_pt > 0, np.log(p_pt + epsilon), 0)
        p_log_m = np.where(p_pt > 0, np.log(p_mass + epsilon), 0)

        p_extra = {}

        lep_btag = np.zeros_like(lep_e)

        if self.include_met:
            met_met = sample["met_met"].values
            met_phi = sample["met_phi"].values

            # Set MET eta and pt to 0 since they are not well-defined for MET
            met_eta = np.zeros_like(met_met)
            met_pt = np.zeros_like(met_met)

            # Calculate MET px and py
            met_px = met_met * np.cos(met_phi)
            met_py = met_met * np.sin(met_phi)

            # Set MET pz to 0 and energy from met_met
            met_pz = np.zeros_like(met_met)
            met_e = met_met  # think this right...need to check

            # Set MET b-tag score to 0, similar to leptons
            met_btag = np.zeros_like(met_met)

            met_theta = np.zeros_like(met_met)

            # set rest mass to zero
            met_mass = np.zeros_like(met_met)

            p_px = np.concatenate((p_px, met_px[:, np.newaxis]), axis=1)
            p_py = np.concatenate((p_py, met_py[:, np.newaxis]), axis=1)
            p_pz = np.concatenate((p_pz, met_pz[:, np.newaxis]), axis=1)
            p_e = np.concatenate((p_e, met_e[:, np.newaxis]), axis=1)
            p_pt = np.concatenate((p_pt, met_pt[:, np.newaxis]), axis=1)
            p_eta = np.concatenate((p_eta, met_eta[:, np.newaxis]), axis=1)
            p_phi = np.concatenate((p_phi, met_phi[:, np.newaxis]), axis=1)
            p_theta = np.concatenate((p_theta, met_theta[:, np.newaxis]), axis=1)
            p_mass = np.concatenate((p_mass, met_mass[:, np.newaxis]), axis=1)

            met_flag = np.ones_like(met_met)
            p_flag = np.concatenate(
                (np.zeros((len(p_pt), self.max_particles)), met_flag[:, np.newaxis]),
                axis=1,
            )

        if self.extra_feats is not None:
            feat = self.extra_feats
            if feat == "btag":
                jet_btag = self.get_jet_features(
                    sample, "jet_tagWeightBin_DL1r_Continuous", max_jets
                )
                # NOT USED YET
                # self.ht_all = sample["HT_all"].values
                # self.njets = sample["nJets"].values
                if self.include_met:
                    p_extra["btag"] = np.concatenate(
                        (lep_btag, jet_btag, met_btag[:, np.newaxis]), axis=-1
                    )
                else:
                    p_extra["btag"] = np.concatenate((lep_btag, jet_btag), axis=-1)
                # NOT USED YET
            # elif feat == ["jet_charge"]:
            #     p_extra["jet_charge"] = self.get_jet_features(sample, "jet_charge", max_jets)
            #     p_extra["jet_charge"] = np.hstack((np.zeros((len(lep_pt), 1)), p_extra["jet_charge"]), axis=-1)
            else:
                logging.error(f"Feature {feat} not recognized! Skipping...")

        four_vectors = None
        # add extra features to the four-vectors
        if self.extra_feats:
            if self.representation:
                four_vectors = np.stack(
                    (p_px, p_py, p_pz, p_e, p_pt, p_eta, p_phi), axis=-1
                )
                p_extra[feat] = np.expand_dims(p_extra[feat], axis=-1)
                four_vectors = np.concatenate((four_vectors, p_extra[feat]), axis=-1)

            elif self.further_representation:
                # more four-vec representations:
                four_vectors = np.stack(
                    (p_px, p_py, p_pz, p_e, p_pt, p_eta, p_phi, p_theta, p_mass),
                    axis=-1,
                )
                # four_vectors_extra = np.stack(( ), axis=-1)
                # p_E_over_m, p_pT_over_m, p_log_pT, p_log_m, np.cos(p_phi), (np.sin(p_phi)),
                p_extra[feat] = np.expand_dims(p_extra[feat], axis=-1)

                four_vectors = np.concatenate((four_vectors, p_extra[feat]), axis=-1)

                # TEMP
                # ht_all_expanded = np.expand_dims(self.ht_all, axis=(1, 2))
                # njets_expanded = np.expand_dims(self.njets, axis=(1, 2))

                # TEMP EXPANSION TO RIGHT DIM -> want this to be global graph feature in the future
                # ht_all_tiled = np.tile(ht_all_expanded, (1, four_vectors.shape[1], 1))
                # njets_tiled = np.tile(njets_expanded, (1, four_vectors.shape[1], 1))
                # four_vectors = np.concatenate((four_vectors, ht_all_tiled), axis=-1)
                # four_vectors = np.concatenate((four_vectors, njets_tiled), axis=-1)

            else:
                p_extra[feat] = np.expand_dims(p_extra[feat], axis=-1)
                four_vectors = np.stack((p_px, p_py, p_pz, p_e), axis=-1)
                four_vectors = np.concatenate((four_vectors, p_extra[feat]), axis=-1)
                # TODO: validate with degbugging logs
        else:
            logging.info("FeatureFactory :: Using only the four-vectors in the model.")
            four_vectors = np.stack((p_px, p_py, p_pz, p_e), axis=-1)

        if self.include_met:
            four_vectors = np.concatenate(
                (four_vectors, p_flag[:, :, np.newaxis]), axis=-1
            )

        # self.log_features(four_vectors, lep_pt, lep_eta, lep_phi, lep_e, jet_pt, jet_eta, jet_phi, jet_e,
        #   p_pt, p_eta, p_phi, p_e, p_px, p_py, p_pz, p_theta, p_mass, p_E_over_m, p_pT_over_m,
        #   p_log_pT, p_log_m)

        return four_vectors

    def construct_edges(self, sample, max_distance=3.5):
        """Constructs the edges between particles (jets and leptons) in each event of the given sample."""
        start_time = time.time()
        p_px, p_py, p_pz, p_e = self._get_basic_four_vectors(sample)
        edges_list, edge_attr_list = self._get_angular_separation(
            sample, p_px, p_py, p_pz, p_e, max_distance
        )

        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(
            f"FeatureFactory :: Constructing egdes -> Execution time: {execution_time:.2f} seconds"
        )

        return edges_list, edge_attr_list

    def _get_angular_separation(self, sample, p_px, p_py, p_pz, p_e, max_distance=3.5):
        """Computes the angular separation (deltaR) and invariant mass between particles (jets and leptons)
        in each event of the given sample and returns the edges and edge attributes."""
        num_events = sample.shape[0]
        edges_list = []
        edge_attr_list = []

        # precompute jet and lepton features for all events
        jet_eta = self.get_jet_features(sample, "jet_eta", self.max_particles).astype(
            np.float64
        )
        jet_phi = self.get_jet_features(sample, "jet_phi", self.max_particles).astype(
            np.float64
        )
        lep_eta = self.compute_lepton_arrays(sample, "eta")
        lep_phi = self.compute_lepton_arrays(sample, "phi")

        # jet and lepton data for all events
        particle_eta = np.concatenate((jet_eta, lep_eta), axis=1)
        particle_phi = np.concatenate((jet_phi, lep_phi), axis=1)

        for event_idx in range(num_events):
            event_eta = particle_eta[event_idx]
            event_phi = particle_phi[event_idx]

            event_px = p_px[event_idx]
            event_py = p_py[event_idx]
            event_pz = p_pz[event_idx]
            event_e = p_e[event_idx]

            event_edges, event_edge_attr = self._compute_event_edges(
                event_px,
                event_py,
                event_pz,
                event_e,
                event_idx,
                event_eta,
                event_phi,
                max_distance,
            )

            edges_list.append(event_edges)
            edge_attr_list.append(event_edge_attr)

        return edges_list, edge_attr_list

    def _compute_event_edges(
        self,
        event_px,
        event_py,
        event_pz,
        event_e,
        event_idx,
        event_eta,
        event_phi,
        max_distance,
    ):
        """Computes the edges and edge attributes for a single event."""
        # KDTree for nearest neighbor search
        points = np.column_stack((event_eta, event_phi))
        tree = cKDTree(points)

        # query the tree for pairs of points within the maximum distance
        event_edges = tree.query_pairs(max_distance)
        event_edges = np.array(list(event_edges)).T

        # we need to make sure we only use valid edges for the inv. mass calc
        valid_edges = (event_edges[0] < event_px.shape[0]) & (
            event_edges[1] < event_px.shape[0]
        )
        event_edges = event_edges[:, valid_edges]

        if event_edges.size == 0:
            event_edges = np.empty((2, 0), dtype=int)
            event_edge_attr = np.empty((0, 6), dtype=float)
        else:
            event_edge_attr = self._compute_edge_attributes(
                event_px,
                event_py,
                event_pz,
                event_e,
                event_idx,
                event_edges,
                event_eta,
                event_phi,
            )

        return event_edges, event_edge_attr

    def _compute_edge_attributes(
        self,
        event_px,
        event_py,
        event_pz,
        event_e,
        event_idx,
        event_edges,
        event_eta,
        event_phi,
    ):
        """Computes the edge attributes (angular separation and invariant mass) for a single event."""
        delta_eta = event_eta[event_edges[0]] - event_eta[event_edges[1]]
        delta_phi = (
            np.mod(
                event_phi[event_edges[0]] - event_phi[event_edges[1]] + np.pi, 2 * np.pi
            )
            - np.pi
        )
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        cos_delta_R = np.cos(delta_R)
        sin_delta_R = np.sin(delta_R)

        p1_px = event_px[event_edges[0]]
        p1_py = event_py[event_edges[0]]
        p1_pz = event_pz[event_edges[0]]
        p1_e = event_e[event_edges[0]]

        p2_px = event_px[event_edges[1]]
        p2_py = event_py[event_edges[1]]
        p2_pz = event_pz[event_edges[1]]
        p2_e = event_e[event_edges[1]]

        invariant_mass = np.sqrt(
            np.maximum(
                0, 2 * (p1_e * p2_e - (p1_px * p2_px + p1_py * p2_py + p1_pz * p2_pz))
            )
        )

        invariant_mass = invariant_mass * 0.001  # convert to GeV

        event_edge_attr = np.column_stack(
            (delta_eta, delta_phi, delta_R, cos_delta_R, sin_delta_R, invariant_mass)
        )
        return event_edge_attr


    def get_matched_objetcs():
        return None  # TODO: implement method (for the Higgs decay angles and other matching tasks)
        # once we have the matched objects, boost to relevant rest frame
        # and calculate decay angles of the Higgs, top,W's, and gluons
        # then save the decay angles as extra features and concat to the four-vectors
        # could also add the jet substructure variables and energy flow correlations after
        # the decay angles, alongside some invariant mass variables?
        # add met to extra variables as well, with flag for missing met

    # TODO: fix and implement
    def log_features(
        self,
        four_vectors,
        lep_pt,
        lep_eta,
        lep_phi,
        lep_e,
        jet_pt,
        jet_eta,
        jet_phi,
        jet_e,
        p_pt,
        p_eta,
        p_phi,
        p_e,
        p_px,
        p_py,
        p_pz,
        p_theta,
        p_mass,
        p_E_over_m,
        p_pT_over_m,
        p_log_pT,
        p_log_m,
        verbose=False,
    ):
        """Log the features extracted from the input samples with optional verbosity using tabulated format.

        Parameters
        ----------
        four_vectors : numpy.ndarray
            The extracted four-vectors.
        lep_pt, lep_eta, lep_phi, lep_e : numpy.ndarray
            The lepton quantities.
        jet_pt, jet_eta, jet_phi, jet_e : numpy.ndarray
            The jet quantities.
        p_pt, p_eta, p_phi, p_e, p_px, p_py, p_pz, p_theta, p_mass, p_E_over_m, p_pT_over_m, p_log_pT, p_log_m : numpy.ndarray
            The combined quantities.
        verbose : bool, optional
            If True, logs detailed data. Defaults to False.
        """

        logging.info("FeatureFactory :: Logging the extracted features...")
        logging.info(
            f"FeatureFactory :: Four-vectors shape: {four_vectors.shape}, dtype: {four_vectors.dtype}"
        )

        def log_table(data):
            headers = ["Statistic", "Value"]
            table = tabulate(data, headers, tablefmt="pretty")
            logging.info("\n" + table)

        def log_summary(name, data):
            summary_data = [
                ["mean", np.mean(data)],
                ["min", np.min(data)],
                ["max", np.max(data)],
                ["std", np.std(data)],
            ]
            logging.info(f"Statistics for {name}:")
            log_table(summary_data)

        if verbose:
            logging.info("Detailed data logging enabled.")

            logging.info("Lepton Quantities:")
            for name, array in zip(
                ["lep_pt", "lep_eta", "lep_phi", "lep_e"],
                [lep_pt, lep_eta, lep_phi, lep_e],
            ):
                log_summary(name, array)

            logging.info("Jet Quantities:")
            for name, array in zip(
                ["jet_pt", "jet_eta", "jet_phi", "jet_e"],
                [jet_pt, jet_eta, jet_phi, jet_e],
            ):
                log_summary(name, array)

            logging.info("Combined Quantities:")
            for name, array in zip(
                [
                    "p_pt",
                    "p_eta",
                    "p_phi",
                    "p_e",
                    "p_px",
                    "p_py",
                    "p_pz",
                    "p_theta",
                    "p_mass",
                    "p_E_over_m",
                    "p_pT_over_m",
                    "p_log_pT",
                    "p_log_m",
                ],
                [
                    p_pt,
                    p_eta,
                    p_phi,
                    p_e,
                    p_px,
                    p_py,
                    p_pz,
                    p_theta,
                    p_mass,
                    p_E_over_m,
                    p_pT_over_m,
                    p_log_pT,
                    p_log_m,
                ],
            ):
                log_summary(name, array)
        else:
            logging.info("Summary data logging only.")
            combined_data = np.concatenate(
                [
                    lep_pt,
                    lep_eta,
                    lep_phi,
                    lep_e,
                    jet_pt,
                    jet_eta,
                    jet_phi,
                    jet_e,
                    p_pt,
                    p_eta,
                    p_phi,
                    p_e,
                    p_px,
                    p_py,
                    p_pz,
                    p_theta,
                    p_mass,
                    p_E_over_m,
                    p_pT_over_m,
                    p_log_pT,
                    p_log_m,
                ]
            )
            log_summary("All Quantities Combined", combined_data)


"""
TODO:
    - Higgs decay angles (use bitwise operations of truth matching) (old CMS BDT for ttH)
        - boost to rest frame of Higgs
        - boost to rest frame of top (and take left-over)
    - Jet substructure variables
    - Energy flow correlations (no clue: https://arxiv.org/pdf/1305.0007.pdf;
      https://jduarte.physics.ucsd.edu/iaifi-summer-school/1.1_tabular_data_efps.html)
    - 2neutrino scanning method (https://indico.cern.ch/event/1032082/#2-dilepton-ttbar-reconstruction)
"""
