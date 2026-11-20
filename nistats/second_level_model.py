"""
This module presents an interface to use the glm implemented in
nistats.regression.

It provides facilities to realize a second level analysis on lists of
first level contrasts or directly on fitted first level models

Author: Martin Perez-Guevara, 2016
"""

import sys
import time
from warnings import warn

import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import CacheMixin
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
from nilearn.mass_univariate import permuted_ols
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.externals.joblib import Memory

from .first_level_model import FirstLevelModel
from .first_level_model import run_glm
from .regression import OLSModel, SimpleRegressionResults
from .contrasts import (Contrast,
                        compute_contrast,
                        expression_to_contrast_vector,
                        )
from .utils import _basestring
from .design_matrix import make_second_level_design_matrix
from nistats._utils.helpers import replace_parameters


#: Noise models accepted by SecondLevelModel
#: 'ols' treats every image as an independent observation.
#: 'cluster' aggregates residual scores within subject groups and uses a
#: finite-sample-corrected cluster-robust covariance.
_SECOND_LEVEL_NOISE_MODELS = ['ols', 'cluster']


def _check_second_level_input(second_level_input, design_matrix,
                              confounds=None, flm_object=True, df_object=True):
    """Checking second_level_input type"""
    # Check parameters
    # check first level input
    if isinstance(second_level_input, list):
        if len(second_level_input) < 2:
            raise ValueError('A second level model requires a list with at'
                             ' least two first level models or niimgs')
        # Check FirstLevelModel objects case
        if flm_object and isinstance(second_level_input[0], FirstLevelModel):
            models_input = enumerate(second_level_input)
            for model_idx, first_level_model in models_input:
                if (
                    first_level_model.labels_ is None or
                    first_level_model.results_ is None
                ):
                    raise ValueError(
                        'Model %s at index %i has not been fit yet'
                        '' % (first_level_model.subject_label, model_idx))
                if not isinstance(first_level_model, FirstLevelModel):
                    raise ValueError(' object at idx %d is %s instead of'
                                     ' FirstLevelModel object' %
                                     (model_idx, type(first_level_model)))
                if confounds is not None:
                    if first_level_model.subject_label is None:
                        raise ValueError(
                            'In case confounds are provided, first level '
                            'objects need to provide the attribute '
                            'subject_label to match rows appropriately.'
                            'Model at idx %d does not provide it. '
                            'To set it, you can do '
                            'first_level_model.subject_label = "01"'
                            '' % (model_idx))
        # Check niimgs case
        elif isinstance(second_level_input[0], (str, Nifti1Image)):
            if design_matrix is None:
                raise ValueError('List of niimgs as second_level_input'
                                 ' require a design matrix to be provided')
            for model_idx, niimg in enumerate(second_level_input):
                if not isinstance(niimg, (str, Nifti1Image)):
                    raise ValueError(' object at idx %d is %s instead of'
                                     ' Niimg-like object' %
                                     (model_idx, type(niimg)))
    # Check pandas dataframe case
    elif df_object and isinstance(second_level_input, pd.DataFrame):
        for col in ['subject_label', 'map_name', 'effects_map_path']:
            if col not in second_level_input.columns:
                raise ValueError('second_level_input DataFrame must have'
                                 ' columns subject_label, map_name and'
                                 ' effects_map_path')
        # Make sure subject_label contain strings
        second_level_columns = second_level_input.columns.tolist()
        labels_index = second_level_columns.index('subject_label')
        labels_dtype = second_level_input.dtypes[labels_index]
        if not isinstance(labels_dtype, np.object):
            raise ValueError('subject_label column must be of dtype '
                             'object instead of dtype %s' % labels_dtype)
    elif isinstance(second_level_input, (str, Nifti1Image)):
        if design_matrix is None:
            raise ValueError('List of niimgs as second_level_input'
                             ' require a design matrix to be provided')
        second_level_input = check_niimg(niimg=second_level_input,
                                         ensure_ndim=4)
    else:
        if flm_object and df_object:
            raise ValueError('second_level_input must be a list of'
                             ' `FirstLevelModel` objects, a pandas DataFrame'
                             ' or a list Niimg-like objects. Instead %s '
                             'was provided' % type(second_level_input))
        else:
            raise ValueError('second_level_input must be'
                             ' a list Niimg-like objects. Instead %s '
                             'was provided' % type(second_level_input))


def _check_confounds(confounds):
    """Checking confounds type"""
    if confounds is not None:
        if not isinstance(confounds, pd.DataFrame):
            raise ValueError('confounds must be a pandas DataFrame')
        if 'subject_label' not in confounds.columns:
            raise ValueError('confounds DataFrame must contain column'
                             ' "subject_label"')
        if len(confounds.columns) < 2:
            raise ValueError('confounds should contain at least 2 columns'
                             ' one called "subject_label" and the other'
                             ' with a given confound')
        # Make sure subject_label contain strings
        labels_index = confounds.columns.tolist().index('subject_label')
        labels_dtype = confounds.dtypes[labels_index]
        if not isinstance(labels_dtype, np.object):
            raise ValueError('subject_label column must be of dtype '
                             'object instead of dtype %s' % labels_dtype)


def _check_first_level_contrast(second_level_input, first_level_contrast):
    if isinstance(second_level_input, list) and isinstance(
            second_level_input[0], FirstLevelModel):
        if first_level_contrast is None:
            raise ValueError('If second_level_input was a list of '
                             'FirstLevelModel, then first_level_contrast '
                             'is mandatory. It corresponds to the '
                             'second_level_contrast argument of the '
                             'compute_contrast method of FirstLevelModel')


def _check_output_type(output_type, valid_types):
    if output_type not in valid_types:
        raise ValueError('output_type must be one of {}'.format(valid_types))


def _check_design_matrix(design_matrix):
    """Checking design_matrix type"""
    if design_matrix is not None:
        if not isinstance(design_matrix, pd.DataFrame):
            raise ValueError('design matrix must be a pandas DataFrame')


def _check_effect_maps(effect_maps, design_matrix):
    if len(effect_maps) != design_matrix.shape[0]:
        raise ValueError(
            'design_matrix does not match the number of maps considered. '
            '%i rows in design matrix do not match with %i maps' %
            (design_matrix.shape[0], len(effect_maps)))


def _get_con_val(second_level_contrast, design_matrix):
    """ Check the contrast and return con_val when testing one contrast or more
    """
    if second_level_contrast is None:
        if design_matrix.shape[1] == 1:
            second_level_contrast = np.ones([1])
        else:
            raise ValueError('No second-level contrast is specified.')
    if not isinstance(second_level_contrast, _basestring):
        con_val = second_level_contrast
        if np.all(con_val == 0):
            raise ValueError('Contrast is null')
    else:
        design_columns = design_matrix.columns.tolist()
        con_val = expression_to_contrast_vector(
            second_level_contrast, design_columns)
    return con_val


def _get_contrast(second_level_contrast, design_matrix):
    """ Check and return contrast when testing one contrast at the time """
    if isinstance(second_level_contrast, str):
        if second_level_contrast in design_matrix.columns.tolist():
            contrast = second_level_contrast
        else:
            raise ValueError('"' + second_level_contrast + '" is not a valid' +
                             ' contrast name')
    else:
        # Check contrast definition
        if second_level_contrast is None:
            if design_matrix.shape[1] == 1:
                second_level_contrast = np.ones([1])
            else:
                raise ValueError('No second-level contrast is specified.')
        elif (np.nonzero(second_level_contrast)[0]).size != 1:
            raise ValueError('second_level_contrast must be '
                             'a list of 0s and 1s')
        con_val = np.asarray(second_level_contrast, dtype=bool)
        contrast = np.asarray(design_matrix.columns.tolist())[con_val][0]
    return contrast


def _infer_effect_maps(second_level_input, contrast_def):
    """Deals with the different possibilities of second_level_input"""
    # Build the design matrix X and list of imgs Y for GLM fit
    if isinstance(second_level_input, pd.DataFrame):
        # If a Dataframe was given, we expect contrast_def to be in map_name
        def _is_contrast_def(x):
            return x['map_name'] == contrast_def

        is_con = second_level_input.apply(_is_contrast_def, axis=1)
        effect_maps = second_level_input[is_con]['effects_map_path'].tolist()

    elif isinstance(second_level_input[0], FirstLevelModel):
        # Get the first level model maps
        effect_maps = []
        for model in second_level_input:
            effect_map = model.compute_contrast(contrast_def,
                                                output_type='effect_size')
            effect_maps.append(effect_map)

    else:
        effect_maps = second_level_input

    # check niimgs
    for niimg in effect_maps:
        check_niimg(niimg, ensure_ndim=3)

    return effect_maps


def _check_design_estimability(design_matrix):
    """Check that the design matrix has full column rank, i.e. that the
    fixed effects are estimable. Raise otherwise.
    """
    X = np.asarray(design_matrix.values, dtype=np.float64)
    eps = np.abs(X).sum() * np.finfo(np.float64).eps
    rank = matrix_rank(X, eps)
    if rank < X.shape[1]:
        raise ValueError(
            'The design matrix is not full rank (rank %i for %i columns), '
            'so the fixed effects are not estimable. Remove collinear '
            'regressors before fitting the second level model.'
            % (rank, X.shape[1]))


def _infer_subject_groups(second_level_input):
    """Return one group label per image row, aligned with the effect maps
    that ``_infer_effect_maps`` will produce.

    A pandas DataFrame is sorted by subject_label in ``fit`` before the
    model is stored, so its subject_label column is already aligned with
    the image rows. A list of FirstLevelModel objects exposes one
    ``subject_label`` per model.
    """
    if isinstance(second_level_input, pd.DataFrame):
        groups = second_level_input['subject_label'].values
    elif isinstance(second_level_input, list) and isinstance(
            second_level_input[0], FirstLevelModel):
        groups = np.asarray(
            [model.subject_label for model in second_level_input])
    else:
        groups = None
    return groups


def _check_subject_groups(groups, n_rows, noise_model):
    """Validate the subject grouping used by the cluster-robust model.

    Ensures that group labels exist, are aligned with the image rows and
    define enough independent groups for a finite-sample correction.
    """
    if noise_model != 'cluster':
        return
    if groups is None:
        raise ValueError(
            "noise_model='cluster' requires subject group labels. "
            'Pass them through the groups argument of fit, or use a '
            'second_level_input that carries subject labels '
            '(pandas DataFrame or list of FirstLevelModel objects).')
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError('groups must be a one-dimensional sequence of '
                         'subject labels, got an array with %i dimensions'
                         % groups.ndim)
    if groups.shape[0] != n_rows:
        raise ValueError(
            'groups must contain one label per image row: %i labels '
            'provided for %i image rows. Group labels must be aligned '
            'with the rows of the design matrix and the effect maps.'
            % (groups.shape[0], n_rows))
    unique_groups, counts = np.unique(groups, return_counts=True)
    n_groups = unique_groups.size
    if n_groups < 2:
        raise ValueError(
            'A cluster-robust covariance requires at least two independent '
            'subject groups to estimate variability; got %i group. Repeated '
            'measurements within a single subject do not provide any '
            'between-subject information.' % n_groups)
    if np.any(counts == 0):
        # np.unique never returns absent labels, this is defensive
        raise ValueError('groups contains empty groups')
    return groups, unique_groups


def _cluster_design_quantities(X, groups):
    """Pre-compute the fixed-design quantities shared by every voxel for
    the finite-sample-corrected cluster-robust (CR2) covariance.

    Parameters
    ----------
    X : array of shape (n_observations, n_regressors)
        The fixed effects design matrix, assumed full column rank.

    groups : array of shape (n_observations,)
        Subject group label of each image row.

    Returns
    -------
    cluster_info : list of dict
        One entry per unique group with keys:
        'indices': row indices of the group,
        'adjustment': array of shape (n_g, n_g), the symmetric inverse
        square root of (I_g - X_g (X'X)^-1 X_g'), the cluster leverage
        adjustment used in the CR2 correction.

    bread : array of shape (p, p)
        (X'X)^-1, the OLS "bread" matrix.

    References
    ----------
    Bell, R. M. and McCaffrey, D. F. (2002). Bias reduction in standard
    errors for linear regression with multiple-stage samples. Survey
    Methodology, 28(2), 169-181.
    """
    X = np.asarray(X, dtype=np.float64)
    xtx_inv = np.linalg.inv(np.dot(X.T, X))
    unique_groups = np.unique(groups)
    cluster_info = []
    for group in unique_groups:
        indices = np.where(groups == group)[0]
        x_g = X[indices]
        # Cluster hat matrix block H_gg = X_g (X'X)^-1 X_g'
        hat_g = np.dot(x_g, np.dot(xtx_inv, x_g.T))
        # I_g - H_gg is symmetric; symmetrize to kill rounding asymmetry
        adjustment_matrix = np.eye(indices.size) - hat_g
        adjustment_matrix = (adjustment_matrix + adjustment_matrix.T) / 2.
        # Eigen-decomposition based symmetric inverse square root,
        # robust against tiny negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(adjustment_matrix)
        eigenvalues = np.maximum(eigenvalues,
                                 np.finfo(np.float64).eps)
        adjustment = np.dot(
            eigenvectors,
            np.dot(np.diag(1. / np.sqrt(eigenvalues)), eigenvectors.T))
        cluster_info.append({'indices': indices, 'adjustment': adjustment})
    return cluster_info, xtx_inv


def _cluster_robust_contrast_variance(Y, X, beta, con_val, cluster_info,
                                      xtx_inv):
    """CR2 cluster-robust variance of a single-row t contrast, voxelwise.

    The OLS point estimates ``beta`` are kept (the fixed effects design is
    fitted unchanged). Residual score vectors X_i' e_i are summed within
    each subject group, corrected with the finite-sample leverage
    adjustment, and combined in a sandwich covariance. The contrast
    variance c' V c is returned directly to avoid storing a full
    (p, p, n_voxels) matrix.

    Parameters
    ----------
    Y : array of shape (n_observations, n_voxels)
        Masked effect maps.

    X : array of shape (n_observations, n_regressors)
        Design matrix.

    beta : array of shape (n_regressors, n_voxels)
        OLS coefficient estimates.

    con_val : array of shape (n_regressors,)
        Single t contrast vector.

    cluster_info, xtx_inv : outputs of ``_cluster_design_quantities``

    Returns
    -------
    variance : array of shape (n_voxels,)
        CR2 cluster-robust estimate of Var(c' beta) at each voxel.
    """
    residuals = Y - np.dot(X, beta)
    meat = 0.
    for info in cluster_info:
        indices = info['indices']
        # CR2-corrected residual score of the group:
        # X_g' (I_g - H_gg)^(-1/2) e_g
        adjusted_residuals = np.dot(info['adjustment'],
                                    residuals[indices])
        score_g = np.dot(X[indices].T, adjusted_residuals)
        # accumulate score_g score_g' on the contrast metric, without
        # building the (p, p, n_voxels) sandwich matrix
        weighted_score = np.dot(con_val, np.dot(xtx_inv, score_g))
        meat += weighted_score ** 2
    return meat


class SecondLevelModel(BaseEstimator, TransformerMixin, CacheMixin):
    """ Implementation of the General Linear Model for multiple subject
    fMRI data

    Parameters
    ----------

    mask_img: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional,
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters. Automatic mask computation assumes first level imgs have
        already been masked.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    memory: string, optional
        Path to the directory used to cache the masking process and the glm
        fit. By default, no caching is done. Creates instance of joblib.Memory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        If 0 prints nothing. If 1 prints final computation time.
        If 2 prints masker computation details.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    minimize_memory : boolean, optional
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption. True by default.

    noise_model : {'ols', 'cluster'}, optional
        Variance model used when computing contrasts.

        If 'ols' (default), every image row is treated as an independent
        observation, as in a standard ordinary least squares second level
        GLM. This is anti-conservative when a subject contributes several
        repeated measurement images, because the within-subject
        observations are correlated.

        If 'cluster', a cluster-robust covariance is used: the fixed
        effects design is still fitted with OLS, but residual score
        vectors (X_i' e_i) are summed within the subject groups given by
        the ``groups`` argument of ``fit`` and combined into a sandwich
        covariance with the CR2 finite-sample correction (Bell and
        McCaffrey, 2002). Contrast statistics then follow a Student t
        distribution with G - 1 degrees of freedom, where G is the number
        of independent subject groups, instead of n - p.

    """

    @replace_parameters({'mask': 'mask_img'}, end_version='next')
    def __init__(self, mask_img=None, smoothing_fwhm=None,
                 memory=Memory(None), memory_level=1, verbose=0,
                 n_jobs=1, minimize_memory=True, noise_model='ols'):
        if noise_model not in _SECOND_LEVEL_NOISE_MODELS:
            raise ValueError(
                'Acceptable noise models are {0}. You provided '
                "'noise_model={1}'".format(_SECOND_LEVEL_NOISE_MODELS,
                                           noise_model))
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        if isinstance(memory, _basestring):
            self.memory = Memory(memory)
        else:
            self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory
        self.noise_model = noise_model
        self.second_level_input_ = None
        self.confounds_ = None
        self.groups_ = None
        self.n_groups_ = None
        self.df_residuals_ = None
        self._cluster_info = None
        self._xtx_inv = None

    def fit(self, second_level_input, confounds=None, design_matrix=None,
            groups=None):
        """ Fit the second-level GLM

        1. create design matrix
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        second_level_input: list of `FirstLevelModel` objects or pandas
                            DataFrame or list of Niimg-like objects.

            Giving FirstLevelModel objects will allow to easily compute
            the second level contast of arbitrary first level contrasts thanks
            to the first_level_contrast argument of the compute_contrast
            method. Effect size images will be computed for each model to
            contrast at the second level.

            If a pandas DataFrame, then they have to contain subject_label,
            map_name and effects_map_path. It can contain multiple maps that
            would be selected during contrast estimation with the argument
            first_level_contrast of the compute_contrast function. The
            DataFrame will be sorted based on the subject_label column to avoid
            order inconsistencies when extracting the maps. So the rows of the
            automatically computed design matrix, if not provided, will
            correspond to the sorted subject_label column.

            If list of Niimg-like objects then this is taken literally as Y
            for the model fit and design_matrix must be provided.

        confounds: pandas DataFrame, optional
            Must contain a subject_label column. All other columns are
            considered as confounds and included in the model. If
            design_matrix is provided then this argument is ignored.
            The resulting second level design matrix uses the same column
            names as in the given DataFrame for confounds. At least two columns
            are expected, "subject_label" and at least one confound.

        design_matrix: pandas DataFrame, optional
            Design matrix to fit the GLM. The number of rows
            in the design matrix must agree with the number of maps derived
            from second_level_input.
            Ensure that the order of maps given by a second_level_input
            list of Niimgs matches the order of the rows in the design matrix.

        groups : array-like of shape (n_images,), optional
            Subject label of each image row, used only when
            ``noise_model='cluster'``. Repeated measurements of the same
            subject must carry the same label. The labels must be aligned
            with the rows of the design matrix and with the effect maps.
            If None, labels are inferred from the ``subject_label`` column
            of a pandas DataFrame input or from the ``subject_label``
            attribute of FirstLevelModel inputs; for a plain list of
            Niimg-like objects, ``groups`` is mandatory in cluster mode.

        """
        # check second_level_input
        _check_second_level_input(second_level_input, design_matrix,
                                  confounds=confounds)

        # check confounds
        _check_confounds(confounds)

        # check design matrix
        _check_design_matrix(design_matrix)

        # sort a pandas dataframe by subject_label to avoid inconsistencies
        # with the design matrix row order when automatically extracting maps
        if isinstance(second_level_input, pd.DataFrame):
            columns = second_level_input.columns.tolist()
            column_index = columns.index('subject_label')
            sorted_matrix = sorted(
                second_level_input.values, key=lambda x: x[column_index])
            sorted_input = pd.DataFrame(sorted_matrix, columns=columns)
            second_level_input = sorted_input

        self.second_level_input_ = second_level_input
        self.confounds_ = confounds

        # Report progress
        t0 = time.time()
        if self.verbose > 0:
            sys.stderr.write("Fitting second level model. "
                             "Take a deep breath\r")

        # Select sample map for masker fit and get subjects_label for design
        if isinstance(second_level_input, pd.DataFrame):
            sample_map = second_level_input['effects_map_path'][0]
            labels = second_level_input['subject_label']
            subjects_label = labels.values.tolist()
        elif isinstance(second_level_input, Nifti1Image):
            sample_map = mean_img(second_level_input)
        elif isinstance(second_level_input[0], FirstLevelModel):
            sample_model = second_level_input[0]
            sample_condition = sample_model.design_matrices_[0].columns[0]
            sample_map = sample_model.compute_contrast(
                sample_condition, output_type='effect_size')
            labels = [model.subject_label for model in second_level_input]
            subjects_label = labels
        else:
            # In this case design matrix had to be provided
            sample_map = mean_img(second_level_input)

        # Create and set design matrix, if not given
        if design_matrix is None:
            design_matrix = make_second_level_design_matrix(subjects_label,
                                                            confounds)
        self.design_matrix_ = design_matrix

        # A cluster-robust fit is only meaningful on an estimable design;
        # check rank explicitly instead of silently relying on the
        # pseudoinverse, which would hide collinear fixed effects.
        if self.noise_model == 'cluster':
            _check_design_estimability(design_matrix)
            if groups is None:
                groups = _infer_subject_groups(second_level_input)
            groups, unique_groups = _check_subject_groups(
                groups, design_matrix.shape[0], self.noise_model)
            self.groups_ = groups
            self.n_groups_ = unique_groups.size
            # Degrees of freedom follow the declared independent-group
            # rule: G independent clusters -> G - 1 residual dof
            self.df_residuals_ = self.n_groups_ - 1
            self._cluster_info, self._xtx_inv = \
                _cluster_design_quantities(design_matrix.values, groups)
        else:
            self.groups_ = None
            self.n_groups_ = None
            self.df_residuals_ = None
            self._cluster_info = None
            self._xtx_inv = None

        # Learn the mask. Assume the first level imgs have been masked.
        if not isinstance(self.mask_img, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask_img, smoothing_fwhm=self.smoothing_fwhm,
                memory=self.memory, verbose=max(0, self.verbose - 1),
                memory_level=self.memory_level)
        else:
            self.masker_ = clone(self.mask_img)
            for param_name in ['smoothing_fwhm', 'memory', 'memory_level']:
                our_param = getattr(self, param_name)
                if our_param is None:
                    continue
                if getattr(self.masker_, param_name) is not None:
                    warn('Parameter %s of the masker overriden' % param_name)
                setattr(self.masker_, param_name, our_param)
        self.masker_.fit(sample_map)

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of second level model done in "
                             "%i seconds\n" % (time.time() - t0))

        return self

    def compute_contrast(self, second_level_contrast=None,
                         first_level_contrast=None,
                         second_level_stat_type=None, output_type='z_score'):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.

        Parameters
        ----------
        second_level_contrast: str or array of shape (n_col), optional
            Where ``n_col`` is the number of columns of the design matrix. The
            string can be a formula compatible with `pandas.DataFrame.eval`.
            Basically one can use the name of the conditions as they appear in
            the design matrix of the fitted model combined with operators +-
            and combined with numbers with operators +-`*`/. The default (None)
            is accepted if the design matrix has a single column, in which case
            the only possible contrast array([1]) is applied; when the design
            matrix has multiple columns, an error is raised.

        first_level_contrast: str or array of shape (n_col) with respect to
                              FirstLevelModel, optional

            In case a list of FirstLevelModel was provided as
            second_level_input, we have to provide a contrast to apply to
            the first level models to get the corresponding list of images
            desired, that would be tested at the second level. In case a
            pandas DataFrame was provided as second_level_input this is the
            map name to extract from the pandas dataframe map_name column.
            It has to be a 't' contrast.

        second_level_stat_type: {'t', 'F'}, optional
            Type of the second level contrast. F contrasts are not
            available with ``noise_model='cluster'``.

        output_type: str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            'effect_size', 'effect_variance' or 'all'

        Returns
        -------
        output_image: Nifti1Image
            The desired output image(s). If ``output_type == 'all'``, then
            the output is a dictionary of images, keyed by the type of image.

        """
        if self.second_level_input_ is None:
            raise ValueError('The model has not been fit yet')

        # check first_level_contrast
        _check_first_level_contrast(self.second_level_input_,
                                    first_level_contrast)

        # check contrast and obtain con_val
        con_val = _get_con_val(second_level_contrast, self.design_matrix_)

        # check output type
        # 'all' is assumed to be the final entry;
        # if adding more, place before 'all'
        valid_types = ['z_score', 'stat', 'p_value', 'effect_size',
                       'effect_variance', 'all']
        _check_output_type(output_type, valid_types)

        # Get effect_maps appropriate for chosen contrast
        effect_maps = _infer_effect_maps(self.second_level_input_,
                                         first_level_contrast)
        # Check design matrix X and effect maps Y agree on number of rows
        _check_effect_maps(effect_maps, self.design_matrix_)

        # Fit an Ordinary Least Squares regression for parametric statistics
        Y = self.masker_.transform(effect_maps)
        X = self.design_matrix_.values

        if self.noise_model == 'cluster':
            # The cluster-robust model keeps the OLS point estimates and
            # replaces only the covariance: residual scores are aggregated
            # by subject and corrected for finite samples (CR2).
            if second_level_stat_type == 'F':
                raise ValueError(
                    "F contrasts are not available with "
                    "noise_model='cluster': the cluster-robust covariance "
                    "supports t contrasts only.")
            con_val = np.atleast_1d(np.asarray(con_val, dtype=np.float64))
            if con_val.ndim != 1:
                raise ValueError(
                    "noise_model='cluster' supports one-dimensional t "
                    'contrasts only')
            # Re-derive group labels aligned with the image rows actually
            # selected for this contrast (a DataFrame may hold several
            # maps per subject, filtered by first_level_contrast)
            if isinstance(self.second_level_input_, pd.DataFrame):
                selected_rows = (self.second_level_input_['map_name']
                                 == first_level_contrast)
                groups = self.second_level_input_[selected_rows][
                    'subject_label'].values
            else:
                groups = self.groups_
            # Re-check alignment and enough independent groups against the
            # actual effect maps used here
            groups, unique_groups = _check_subject_groups(
                groups, Y.shape[0], self.noise_model)
            # Recompute the fixed-design quantities against the actual
            # rows of this contrast, so they stay aligned even when the
            # input contains several maps per subject
            cluster_info, xtx_inv = _cluster_design_quantities(X, groups)
            df_residuals = unique_groups.size - 1

            ols_fit = OLSModel(X).fit(Y)
            self.labels_ = np.zeros(Y.shape[1])
            self.results_ = {0.0: ols_fit}
            beta = ols_fit.theta
            effect = np.dot(con_val, beta)
            variance = _cluster_robust_contrast_variance(
                Y, X, beta, con_val, cluster_info, xtx_inv)
            # Native t contrast with the declared independent-group dof
            contrast = Contrast(effect=effect[np.newaxis],
                                variance=variance,
                                dim=1, dof=df_residuals,
                                contrast_type='t')
        else:
            if self.memory:
                mem_glm = self.memory.cache(run_glm, ignore=['n_jobs'])
            else:
                mem_glm = run_glm
            labels, results = mem_glm(Y, X,
                                      n_jobs=self.n_jobs, noise_model='ols')

            # We save memory if inspecting model details is not necessary
            if self.minimize_memory:
                for key in results:
                    results[key] = SimpleRegressionResults(results[key])
            self.labels_ = labels
            self.results_ = results

            # We compute contrast object
            if self.memory:
                mem_contrast = self.memory.cache(compute_contrast)
            else:
                mem_contrast = compute_contrast
            contrast = mem_contrast(self.labels_, self.results_, con_val,
                                    second_level_stat_type)

        output_types = \
            valid_types[:-1] if output_type == 'all' else [output_type]

        outputs = {}
        for output_type_ in output_types:
            # We get desired output from contrast object
            estimate_ = getattr(contrast, output_type_)()
            # Prepare the returned images
            output = self.masker_.inverse_transform(estimate_)
            contrast_name = str(con_val)
            output.header['descrip'] = (
                '%s of contrast %s' % (output_type, contrast_name))
            outputs[output_type_] = output

        return outputs if output_type == 'all' else output


def non_parametric_inference(second_level_input, confounds=None,
                             design_matrix=None, second_level_contrast=None,
                             mask=None, smoothing_fwhm=None,
                             model_intercept=True, n_perm=10000,
                             two_sided_test=False, random_state=None,
                             n_jobs=1, verbose=0):
    """Generate p-values corresponding to the contrasts provided
    based on permutation testing. This fuction reuses the 'permuted_ols'
    function Nilearn.

    Parameters
    ----------
    second_level_input: pandas DataFrame or list of Niimg-like objects.

        If a pandas DataFrame, then they have to contain subject_label,
        map_name and effects_map_path. It can contain multiple maps that
        would be selected during contrast estimation with the argument
        first_level_contrast of the compute_contrast function. The
        DataFrame will be sorted based on the subject_label column to avoid
        order inconsistencies when extracting the maps. So the rows of the
        automatically computed design matrix, if not provided, will
        correspond to the sorted subject_label column.

        If list of Niimg-like objects then this is taken literally as Y
        for the model fit and design_matrix must be provided.

    confounds: pandas DataFrame, optional
        Must contain a subject_label column. All other columns are
        considered as confounds and included in the model. If
        design_matrix is provided then this argument is ignored.
        The resulting second level design matrix uses the same column
        names as in the given DataFrame for confounds. At least two columns
        are expected, "subject_label" and at least one confound.

    design_matrix: pandas DataFrame, optional
        Design matrix to fit the GLM. The number of rows
        in the design matrix must agree with the number of maps derived
        from second_level_input.
        Ensure that the order of maps given by a second_level_input
        list of Niimgs matches the order of the rows in the design matrix.

    second_level_contrast: str or array of shape (n_col), optional
        Where ``n_col`` is the number of columns of the design matrix.
        The default (None) is accepted if the design matrix has a single
        column, in which case the only possible contrast array([1]) is
        applied; when the design matrix has multiple columns, an error is
        raised.

    mask: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional,
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters. Automatic mask computation assumes first level imgs have
        already been masked.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    model_intercept : bool,
      If True, a constant column is added to the confounding variates
      unless the tested variate is already the intercept.

    n_perm : int,
      Number of permutations to perform.
      Permutations are costly but the more are performed, the more precision
      one gets in the p-values estimation.

    two_sided_test : boolean,
      If True, performs an unsigned t-test. Both positive and negative
      effects are considered; the null hypothesis is that the effect is zero.
      If False, only positive effects are considered as relevant. The null
      hypothesis is that the effect is zero or negative.

    random_state : int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    n_jobs : int,
      Number of parallel workers.
      If -1 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (abs(n_jobs) - 1)
      ones will be used.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    neg_log_corrected_pvals_img: Nifti1Image
        The image which contains negative logarithm of the
        corrected p-values
    """
    _check_second_level_input(second_level_input, design_matrix,
                              flm_object=False, df_object=False)
    _check_confounds(confounds)
    _check_design_matrix(design_matrix)

    # Report progress
    t0 = time.time()
    if verbose > 0:
        sys.stderr.write("Fitting second level model...")

    # Select sample map for masker fit and get subjects_label for design
    sample_map = mean_img(second_level_input)

    # Learn the mask. Assume the first level imgs have been masked.
    if not isinstance(mask, NiftiMasker):
        masker = NiftiMasker(
            mask_img=mask, smoothing_fwhm=smoothing_fwhm,
            memory=Memory(None), verbose=max(0, verbose - 1),
            memory_level=1)
    else:
        masker = clone(mask)
        if smoothing_fwhm is not None:
            if getattr(masker, 'smoothing_fwhm') is not None:
                warn('Parameter smoothing_fwhm of the masker overriden')
                setattr(masker, 'smoothing_fwhm', smoothing_fwhm)
    masker.fit(sample_map)

    # Report progress
    if verbose > 0:
        sys.stderr.write("\nComputation of second level model done in "
                         "%i seconds\n" % (time.time() - t0))

    # Check and obtain the contrast
    contrast = _get_contrast(second_level_contrast, design_matrix)

    # Get effect_maps
    effect_maps = _infer_effect_maps(second_level_input, None)

    # Check design matrix and effect maps agree on number of rows
    _check_effect_maps(effect_maps, design_matrix)

    # Obtain tested_var
    if contrast in design_matrix.columns.tolist():
        tested_var = np.asarray(design_matrix[contrast])

    # Mask data
    target_vars = masker.transform(effect_maps)

    # Perform massively univariate analysis with permuted OLS
    neg_log_pvals_permuted_ols, _, _ = permuted_ols(
        tested_var, target_vars, model_intercept=model_intercept,
        n_perm=n_perm, two_sided_test=two_sided_test,
        random_state=random_state, n_jobs=n_jobs,
        verbose=max(0, verbose - 1))
    neg_log_corrected_pvals_img = masker.inverse_transform(
        np.ravel(neg_log_pvals_permuted_ols))

    return neg_log_corrected_pvals_img
