import numpy as np
import logging


class DataAugmentation:
    """Class to perform data augmentation on the input features.

    The class provides methods to apply various types of augmentations to the input features.
    The purpose of this class is to increase the diversity of the training data and help the model
    generalise better to unseen data. The augmentations include:
        - Phi-rotations: Rotate the input features in the phi direction.
        - Eta-reflections: Reflect the input features along the eta axis.
        - Eta-phi translations: Translate the input features in the eta-phi plane.
        - Energy-momentum scaling: Scale the energy and momentum components of the input features.

    These augmentations should only be applied when `use_four_vectors` is set to `True` in the configuration,
    otherwise an error will be raised.

    """
    def __init__(
        self,
        use_four_vectors,
        rotation_range=(0, 2 * np.pi),
        eta_shift_std=0.1,
        phi_shift_std=0.1,
        scale_std=0.1,
    ):
        if not use_four_vectors:
            raise ValueError("Augmentation requires `use_four_vectors` to be True!"
                             "Please set `use_four_vectors` to True in the configuration,"
                             "or disable the data augmentation.")
        """
        Initialise the DataAugmentation class with configuration for augmentations.

        Parameters
        ----------
        rotation_range : tuple, optional
            The range of rotation angles for phi rotation augmentation. Defaults to (0, 2*np.pi).
        eta_shift_std : float, optional
            Standard deviation for normal distribution from which eta shifts are drawn. Defaults to 0.1.
        phi_shift_std : float, optional
            Standard deviation for normal distribution from which phi shifts are drawn. Defaults to 0.1.
        scale_std : float, optional
            Standard deviation for normal distribution from which scaling factors are drawn. Defaults to 0.1.
        """
        self.rotation_range = rotation_range
        self.eta_shift_std = eta_shift_std
        self.phi_shift_std = phi_shift_std
        self.scale_std = scale_std
        self.use_four_vectors = use_four_vectors

        def __str__(self):
            return f"Data augmentation initialised with configuration: {self.__dict__}"

        # log initlialisation of the augmentation
        self.__str__()

    @classmethod
    def standard_augmentation(cls, use_four_vectors):
        """Pre-defined standard augmentation configuration."""
        return cls(
            use_four_vectors,
            rotation_range=(0, 2 * np.pi),
            eta_shift_std=0.1,
            phi_shift_std=0.1,
            scale_std=0.1,
        )

    @classmethod
    def alternative_augmentation(cls, use_four_vectors):
        """Pre-defined alternative augmentation configuration two."""
        return cls(
            use_four_vectors,
            rotation_range=(0, 2 * np.pi),
            eta_shift_std=0.2,
            phi_shift_std=0.2,
            scale_std=0.2,
        )

    def get_number_of_events(self, signal_features, background_features):
        """Get the number of signal and background events in the input features."""
        num_signal_events = signal_features.shape[0]
        num_background_events = background_features.shape[0]
        return num_signal_events, num_background_events

    @staticmethod
    def get_momentum_components(features):
        """Get the 3-momentum components from the input features."""
        px = features[:, :, 0]
        py = features[:, :, 1]
        pz = features[:, :, 2]
        return px, py, pz

    def perform_all_augmentations(self, signal_features, background_features):
        """Perform all augmentations on the input features."""
        augmented_signal_features, augmented_background_features = self.rotate_phi(
            signal_features, background_features
        )
        augmented_signal_features, augmented_background_features = self.reflect_eta(
            augmented_signal_features, augmented_background_features
        )
        augmented_signal_features, augmented_background_features = self.translate_eta_phi(
            augmented_signal_features, augmented_background_features
        )
        augmented_signal_features, augmented_background_features = self.scale_energy_momentum(
            augmented_signal_features, augmented_background_features
        )
        return augmented_signal_features, augmented_background_features

    # ==============================================================================
    # PHI-ROTATIONS

    def rotate_phi(self, signal_features, background_features)-> tuple:
        """
        Apply phi rotation augmentation to the input features.

        This helps the model learn to recognise patterns and features that are
        invariant to rotations in the phi plane. The model becomes more
        robust and can better handle events with different orientations

        Parameters
        ----------
        signal_features : numpy.ndarray
            The extracted signal features (four-vectors and extra features).
        background_features : numpy.ndarray
            The extracted background features (four-vectors and extra features).

        Returns
        -------
        tuple
            A tuple containing the augmented signal and background features.
                augmented_signal_features : numpy.ndarray
                    The augmented signal features.
                augmented_background_features : numpy.ndarray
                    The augmented background features.
        """

        if not self.use_four_vectors:
            raise ValueError("Phi rotation augmentation requires `use_four_vectors` to be True.")

        logging.info("Applying phi-rotations...")
        # get number of events
        num_signal_events, num_background_events = self.get_number_of_events(
            signal_features, background_features
        )

        # angles generated from a uniform distribution
        signal_angles = np.random.uniform(0, 2 * np.pi, size=num_signal_events)
        background_angles = np.random.uniform(0, 2 * np.pi, size=num_background_events)

        # apply rotations to features
        augmented_signal_features = self._rotate_features(
            signal_features, signal_angles
        )
        augmented_background_features = self._rotate_features(
            background_features, background_angles
        )

        logging.info("Phi-rotations complete.")
        return augmented_signal_features, augmented_background_features

    @staticmethod
    def _rotate_features(features, angles):
        """
        Rotate the input features by the given angles in the phi direction.

        Parameters
        ----------
        features : numpy.ndarray
            The input features (four-vectors and extra features).
        angles : numpy.ndarray
            The rotation angles for each event.

        Returns
        -------
        numpy.ndarray
            The rotated features.
        """
        # get px and py components
        px = features[:, :, 0]
        py = features[:, :, 1]

        # now, apply rotations to px and py
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        rotated_px = px * cos_angles[:, np.newaxis] - py * sin_angles[:, np.newaxis]
        rotated_py = px * sin_angles[:, np.newaxis] + py * cos_angles[:, np.newaxis]

        # update the rotated components in the features
        rotated_features = features.copy()
        rotated_features[:, :, 0] = rotated_px
        rotated_features[:, :, 1] = rotated_py


        return rotated_features

    # ==============================================================================
    # ETA-REFLECTIONS

    def reflect_eta(self, signal_features, background_features)-> tuple:
        """
        Apply eta reflection augmentation to the input features.

        Parameters
        ----------
        signal_features : numpy.ndarray
            The extracted signal features (four-vectors and extra features).
        background_features : numpy.ndarray
            The extracted background features (four-vectors and extra features).

        Returns
        -------
        tuple
            A tuple containing the augmented signal and background features.
                augmented_signal_features : numpy.ndarray
                    The augmented signal features.
                augmented_background_features : numpy.ndarray
                    The augmented background features.
        """
        if not self.use_four_vectors:
            raise ValueError("Eta reflection augmentation requires `use_four_vectors` to be True.")

        logging.info("Applying eta-reflections...")
        # generate some random reflection flags for each event
        num_signal_events, num_background_events = self.get_number_of_events(
            signal_features, background_features
        )
        signal_reflect = np.random.choice([True, False], size=num_signal_events)
        background_reflect = np.random.choice([True, False], size=num_background_events)

        # apply eta reflections to the features
        augmented_signal_features = self._reflect_features(
            signal_features, signal_reflect
        )
        augmented_background_features = self._reflect_features(
            background_features, background_reflect
        )

        logging.info("Eta-reflections complete.")
        return augmented_signal_features, augmented_background_features

    @staticmethod
    def _reflect_features(features, reflect):
        """
        Reflect the input features along the eta axis based on the given reflection flags.

        Parameters
        ----------
        features : numpy.ndarray
            The input features (four-vectors and extra features).
        reflect : numpy.ndarray
            The reflection flags for each event.

        Returns
        -------
        numpy.ndarray
            The reflected features.
        """
        # get pz components and reflect using the flags
        pz = features[:, :, 2]
        reflected_pz = np.where(reflect[:, np.newaxis], -pz, pz)

        # update pz components
        reflected_features = features.copy()
        reflected_features[:, :, 2] = reflected_pz

        return reflected_features

    # ==============================================================================
    # ETA-PHI TRANSLATIONS

    def translate_eta_phi(self, signal_features, background_features)-> tuple:
        """
        Apply eta-phi translation augmentation to the input features.

        Parameters
        ----------
        signal_features : numpy.ndarray
            The extracted signal features (four-vectors and extra features).
        background_features : numpy.ndarray
            The extracted background features (four-vectors and extra features).

        Returns
        -------
        tuple
            A tuple containing the augmented signal and background features.
                augmented_signal_features : numpy.ndarray
                    The augmented signal features.
                augmented_background_features : numpy.ndarray
                    The augmented background features.
        """

        if not self.use_four_vectors:
            raise ValueError("Eta-phi translation augmentation requires `use_four_vectors` to be True.")

        logging.info("Applying eta-phi translations...")
        # generate a random shift in eta and phi for each event
        num_signal_events, num_background_events = self.get_number_of_events(
            signal_features, background_features
        )
        signal_eta_shift = np.random.normal(0, 0.1, size=num_signal_events)
        signal_phi_shift = np.random.normal(0, 0.1, size=num_signal_events)
        background_eta_shift = np.random.normal(0, 0.1, size=num_background_events)
        background_phi_shift = np.random.normal(0, 0.1, size=num_background_events)

        # apply translations to the features
        augmented_signal_features = self._translate_features(
            signal_features, signal_eta_shift, signal_phi_shift
        )
        augmented_background_features = self._translate_features(
            background_features, background_eta_shift, background_phi_shift
        )

        logging.info("Eta-phi translations complete.")
        return augmented_signal_features, augmented_background_features

    @staticmethod
    def _translate_features(features, eta_shift, phi_shift):
        """
        Translate the input features in the eta-phi plane based on the given shift values.

        Parameters
        ----------
        features : numpy.ndarray
            The input features (four-vectors and extra features).
        eta_shift : numpy.ndarray
            The eta shift values for each event.
        phi_shift : numpy.ndarray
            The phi shift values for each event.

        Returns
        -------
        numpy.ndarray
            The translated features.
        """
        # extract px, py, and pz components from columns of the arrays
        px, py, pz = DataAugmentation.get_momentum_components(features)

        # calc pt and eta
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz / pt)

        # Apply eta and phi shifts
        translated_eta = eta + eta_shift[:, np.newaxis]
        translated_phi = np.arctan2(py, px) + phi_shift[:, np.newaxis]

        # calc translated px,py,pz
        translated_px = pt * np.cos(translated_phi)
        translated_py = pt * np.sin(translated_phi)
        translated_pz = pt * np.sinh(translated_eta)

        # update components
        translated_features = features.copy()
        translated_features[:, :, 0] = translated_px
        translated_features[:, :, 1] = translated_py
        translated_features[:, :, 2] = translated_pz

        return translated_features

    # ==============================================================================
    # ENERGY-MOMENTUM SCALING

    def scale_energy_momentum(self, signal_features, background_features)-> tuple:
        """
        Apply energy/momentum scaling augmentation to the input features.

        Parameters
        ----------
        signal_features : numpy.ndarray
            The extracted signal features (four-vectors and extra features).
        background_features : numpy.ndarray
            The extracted background features (four-vectors and extra features).

        Returns
        -------
        tuple
            A tuple containing the augmented signal and background features.
                augmented_signal_features : numpy.ndarray
                    The augmented signal features.
                augmented_background_features : numpy.ndarray
                    The augmented background features.
        """
        if not self.use_four_vectors:
            raise ValueError("Energy-momentum scaling augmentation requires `use_four_vectors` to be True.")

        logging.info("Applying energy-momentum scaling...")
        # generate random scaling factors for each event
        num_signal_events, num_background_events = self.get_number_of_events(
            signal_features, background_features
        )
        signal_scales = np.random.normal(1.0, 0.1, size=num_signal_events)
        background_scales = np.random.normal(1.0, 0.1, size=num_background_events)

        # apply scaling to the features
        augmented_signal_features = self._scale_features(signal_features, signal_scales)
        augmented_background_features = self._scale_features(
            background_features, background_scales
        )

        logging.info("Energy-momentum scaling complete.")
        return augmented_signal_features, augmented_background_features

    @staticmethod
    def _scale_features(features, scales):
        """
        Scale the energy and momentum components of the input features by the given factors.

        Parameters
        ----------
        features : numpy.ndarray
            The input features (four-vectors and extra features).
        scales : numpy.ndarray
            The scaling factors for each event.

        Returns
        -------
        numpy.ndarray
            The scaled features.
        """

        # extract px, py, and pz components from columns of the arrays
        px, py, pz = DataAugmentation.get_momentum_components(features)
        energy = features[:, :, 3]

        # multiply by scaling factors
        scaled_px = px * scales[:, np.newaxis]
        scaled_py = py * scales[:, np.newaxis]
        scaled_pz = pz * scales[:, np.newaxis]
        scaled_energy = energy * scales[:, np.newaxis]

        # and update the features with the scaled components
        scaled_features = features.copy()
        scaled_features[:, :, 0] = scaled_px
        scaled_features[:, :, 1] = scaled_py
        scaled_features[:, :, 2] = scaled_pz
        scaled_features[:, :, 3] = scaled_energy

        return scaled_features

