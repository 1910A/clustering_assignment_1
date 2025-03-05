import pandas as pd
from IPython.display import display
import statsmodels.api as sm
import os
# import tempfile
from patsy import dmatrix  # For formula parsing (optional, if you use patsy)
# from typing import Optional
import statsmodels.formula.api as smf
from abc import ABC, abstractmethod
import inspect
import numpy as np

# Trial sequence class and function definitions
def trial_sequence(estimand, **kwargs):
    """
    Factory function to create a TrialSequence object based on the specified estimand.
    
    Args:
        estimand (str): The type of estimand ("ITT", "PP", or "AT").
        **kwargs: Additional arguments passed to the specific TrialSequence subclass.
    
    Returns:
        TrialSequence: An instance of the appropriate TrialSequence subclass.
    
    Raises:
        ValueError: If an invalid estimand is provided.
    """
    estimand_classes = {
        "ITT": TrialSequenceITT,
        "PP": TrialSequencePP,
        "AT": TrialSequenceAT,
    }
    
    if estimand not in estimand_classes:
        raise ValueError(f"Invalid estimand: {estimand}. Choose from ITT, PP, AT.")
    
    return estimand_classes[estimand](**kwargs)

def te_outcome_data(data: pd.DataFrame, p_control: float = None, subset_condition: str = None):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    missing_columns = TEOutcomeData.REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if not pd.api.types.is_numeric_dtype(data["trial_period"]):
        raise ValueError("trial_period column must be numeric")
    n_rows = len(data)
    if n_rows == 0:
        print("Warning: Outcome data has 0 rows")
    n_ids = data["id"].nunique()
    periods = sorted(data["trial_period"].unique())
    subset_condition = subset_condition if subset_condition is not None else ""
    p_control = p_control if p_control is not None else 0.0
    return TEOutcomeData(data, n_rows, n_ids, periods, p_control, subset_condition)

def data_manipulation(data, use_censor=True):
    # Input validation
    if not isinstance(use_censor, bool):
        raise ValueError("use_censor must be a boolean")

    # Copy dataframe to avoid modifying original
    data = data.copy()
    len_data = len(data)
    len_id = data['id'].nunique()

    # Calculate after_eligibility
    data['after_eligibility'] = data.groupby('id').apply(
        lambda x: x['period'] >= x[x['eligible'] == 1]['period'].min() 
        if any(x['eligible'] == 1) else True
    ).reset_index(drop=True)

    if not data['after_eligibility'].all():
        print("Warning: Observations before trial eligibility were removed")
        data = data[data['after_eligibility'] == True]
    
    data = data.drop('after_eligibility', axis=1)

    # Instead of the original after_event calculation, use this:
    data['after_event'] = data.groupby('id').apply(
        lambda x: x['period'] > (
            x[x['outcome'] == 1]['period'].min() 
            if (x['outcome'] == 1).any() 
            else float('inf')
        )
    ).reset_index(drop=True)
    
    # Now this should work without the ValueError
    if data['after_event'].any():
        print("Warning: Observations after the outcome occurred were removed")
        data = data[data['after_event'] == False]
    
    data = data.drop('after_event', axis=1)

    # Calculate event time
    event_data = data.groupby('id').last()[['period', 'outcome']].reset_index()
    event_data['time_of_event'] = 9999
    event_data.loc[(~event_data['outcome'].isna()) & (event_data['outcome'] == 1), 
                  'time_of_event'] = event_data['period'].astype(float)

    # Merge and process switching data
    sw_data = data.merge(event_data[['id', 'time_of_event']], on='id')
    sw_data['first'] = ~sw_data['id'].duplicated()
    sw_data['am_1'] = sw_data.groupby('id')['treatment'].shift()
    
    # Initialize values for first observations
    mask_first = sw_data['first'] == True
    sw_data.loc[mask_first, 'cumA'] = 0
    sw_data.loc[mask_first, 'am_1'] = 0
    sw_data.loc[mask_first, 'switch'] = 0
    sw_data.loc[mask_first, 'regime_start'] = sw_data.loc[mask_first, 'period']
    sw_data.loc[mask_first, 'time_on_regime'] = 0

    # Calculate switch and regime_start for non-first observations
    mask_not_first = sw_data['first'] == False
    sw_data.loc[mask_not_first & (sw_data['am_1'] != sw_data['treatment']), 'switch'] = 1
    sw_data.loc[mask_not_first & (sw_data['am_1'] == sw_data['treatment']), 'switch'] = 0
    sw_data.loc[mask_not_first & (sw_data['switch'] == 1), 'regime_start'] = sw_data['period']
    
    # Forward fill regime_start within groups
    sw_data['regime_start'] = sw_data.groupby('id')['regime_start'].ffill()

    # Calculate time_on_regime
    sw_data['regime_start_shift'] = sw_data.groupby('id')['regime_start'].shift()
    sw_data.loc[mask_not_first, 'time_on_regime'] = (
        sw_data['period'] - sw_data['regime_start_shift'].astype(float)
    )

    # Calculate cumulative treatment
    sw_data.loc[mask_first, 'cumA'] = sw_data.loc[mask_first, 'treatment']
    sw_data.loc[mask_not_first, 'cumA'] = sw_data.loc[mask_not_first, 'treatment']
    sw_data['cumA'] = sw_data.groupby('id')['cumA'].cumsum()
    sw_data = sw_data.drop('regime_start_shift', axis=1)

    # Censoring logic
    if use_censor:
        sw_data['started0'] = np.nan
        sw_data['started1'] = np.nan
        sw_data['stop0'] = np.nan
        sw_data['stop1'] = np.nan
        sw_data['eligible0_sw'] = np.nan
        sw_data['eligible1_sw'] = np.nan
        sw_data['delete'] = np.nan
        
        # Assuming censor_func exists
        sw_data = censor_func(sw_data)
        sw_data = sw_data[sw_data['delete'] == False]
        sw_data = sw_data.drop(['delete', 'eligible0_sw', 'eligible1_sw', 
                              'started0', 'started1', 'stop0', 'stop1'], axis=1)

    # Calculate eligibility indicators
    sw_data['eligible0'] = 0
    sw_data['eligible1'] = 0
    sw_data.loc[sw_data['am_1'] == 0, 'eligible0'] = 1
    sw_data.loc[sw_data['am_1'] == 1, 'eligible1'] = 1

    # Sort by id
    sw_data = sw_data.sort_values('id')
    
    return sw_data

def censor_func(sw_data: pd.DataFrame) -> pd.DataFrame:
    sw_data = sw_data.copy()
    # Initialize columns
    for col in ["started0", "started1", "stop0", "stop1", "eligible0_sw", "eligible1_sw", "delete"]:
        sw_data[col] = 0 if col != "delete" else False

    # First observation per ID
    sw_data["first"] = ~sw_data["id"].duplicated()
    first_mask = sw_data["first"]

    # Reset state at first observation
    for col in ["started0", "started1", "stop0", "stop1", "eligible0_sw", "eligible1_sw"]:
        sw_data.loc[first_mask, col] = 0

    # Vectorized logic for censoring
    eligible = sw_data["eligible"].astype(bool)
    treatment = sw_data["treatment"].astype(int)
    switch = sw_data["switch"].astype(int)

    # Start conditions
    no_start = (sw_data["started0"] == 0) & (sw_data["started1"] == 0) & eligible
    sw_data.loc[no_start & (treatment == 0), "started0"] = 1
    sw_data.loc[no_start & (treatment == 1), "started1"] = 1

    # Eligible switches
    sw_data["eligible0_sw"] = (sw_data["started0"] == 1) & (sw_data["stop0"] == 0)
    sw_data["eligible1_sw"] = (sw_data["started1"] == 1) & (sw_data["stop1"] == 0)

    # Switch logic
    switch_mask = (switch == 1) & eligible
    sw_data.loc[switch_mask & (treatment == 1), ["started1", "eligible1_sw"]] = 1
    sw_data.loc[switch_mask & (treatment == 1), ["started0", "stop0"]] = 0
    sw_data.loc[switch_mask & (treatment == 0), ["started0", "eligible0_sw"]] = 1
    sw_data.loc[switch_mask & (treatment == 0), ["started1", "stop1"]] = 0

    # Stop conditions
    sw_data.loc[switch & ~eligible, "stop0"] = sw_data["started0"]
    sw_data.loc[switch & ~eligible, "stop1"] = sw_data["started1"]

    # Delete rows where neither regime is eligible
    sw_data["delete"] = (sw_data["eligible0_sw"] == 0) & (sw_data["eligible1_sw"] == 0)
    return sw_data

# Helper functions (unchanged from previous)
def update_formula(formula, base="treatment ~ ."):
    if formula.startswith("~"):
        formula = formula[1:]
    return f"treatment ~ {formula}"

def rhs_vars(formula):
    if "~" not in formula:
        return []
    rhs = formula.split("~")[1].strip()
    return [var.strip() for var in rhs.replace("+", " ").split() if var.strip()]

# class StatsGLMLogit(te_model_fitter): # look for its R code in the TrialEmulation package docs because we need to revise this
#     """A simple model fitter for logistic regression using statsmodels."""
#     def __init__(self, save_path=""):
#         super().__init__(save_path=save_path)
#         self.fitted_models = {}

#     def fit_weights_model(self, data, formula):
#         """Fit a logistic regression model for weights."""
#         model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial())
#         result = model.fit()
#         self.fitted_models[formula] = result
#         return result

#     def fit_outcome_model(self):
#         pass  # Not used here, but required by abstract base class

#     def predict(self, data, formula):
#         if formula in self.fitted_models:
#             return self.fitted_models[formula].predict(data)
#         raise ValueError("Model not fitted for this formula")

class TEOutcomeData: # this is the te_outcome_data class in the R TrialEmulation package docs
    REQUIRED_COLUMNS = {"id", "trial_period", "followup_time", "outcome", "weight"}
    
    def __init__(self, data: pd.DataFrame, n_rows: int, n_ids: int, periods: list, p_control: float, subset_condition: str):
        self.data = data
        self.n_rows = n_rows
        self.n_ids = n_ids
        self.periods = periods
        self.p_control = p_control
        self.subset_condition = subset_condition
        self.validate()
    
    def validate(self):
        missing_columns = self.REQUIRED_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def show(self):
        if self.data.empty:
            print("No outcome data, use load_expanded_data()")
        else:
            print("Outcome data")
            print(f"N: {self.n_rows} observations from {self.n_ids} patients")
            print(f"Trial periods: {self.periods} ({len(self.periods)} periods)")
            if self.subset_condition:
                print(f"Subset condition: {self.subset_condition}")
            if self.p_control is not None:
                print(f"Sampling control observations with probability: {self.p_control}")
            display(self.data)
        
    def __repr__(self):
        return (f"TEOutcomeData(n_rows={self.n_rows}, n_ids={self.n_ids}, periods={self.periods}, "
                f"p_control={self.p_control}, subset_condition='{self.subset_condition}')")

class te_data:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.nobs = len(data)
        self.n = data['id'].nunique() if 'id' in data.columns else None
        
        # Validate columns
        required_cols = {"id", "period", "treatment", "outcome", "eligible"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def show(self):
        print(f" - N: {self.nobs} observations from {self.n} patients")
        
        # Hide derived columns except "time_on_regime"
        hide_cols = {"time_of_event", "first", "am_1", "cumA", "switch", "regime_start",
                     "eligible0", "eligible1", "p_n", "p_d", "pC_n", "pC_d"}
        show_cols = [col for col in self.data.columns if col not in hide_cols]
        
        display(self.data[show_cols])  # Show only first 4 rows

class te_data_unset(te_data):
    def __init__(self):
        required_columns = ["id", "period", "treatment", "outcome", "eligible"]
        super().__init__(pd.DataFrame(columns=required_columns))
        # self.nobs = 0
        # self.n = 0
    
    def show(self):
        print(f" - No data has been set. Use set_data()")


class te_datastore:
    def __init__(self, N: int = 0):
        self.N = N

class te_datastore_datatable(te_datastore):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def show(self):
        print("A TE Datastore Datatable object")
        print(f"N: {self.N} observations")
        print(self.data.head(4))  # Show first 4 rows
        
class te_expansion:
    def __init__(self, chunk_size, datastore, censor_at_switch, first_period, last_period):
        self.chunk_size = 0 if chunk_size is None else chunk_size# datatype "numeric",
        self.datastore = te_datastore() if datastore is None else datastore # datatype "te_datastore",
        self.censor_at_switch = censor_at_switch # datatype "logical",
        self.first_period = first_period # datatype "numeric",
        self.last_period = last_period # datatype "numeric"

    def show(self):
        print("Sequence of Trials Data:")
        print(f"- Chunk size: {self.chunk_size}")
        print(f"- Censor at switch: {self.censor_at_switch}")
        print(f"- First period: {self.first_period} | Last period: {self.last_period}")
        
        if (self.datastore.N > 0):
          print("")
          self.datastore.show()
        else:
          print("- Use expand_trials() to construct sequence of trials dataset.")
        
class te_expansion_unset(te_expansion):
    def __init__(self):
        # Get parameter names of parent class __init__
        params = inspect.signature(te_expansion.__init__).parameters
        
        # Dynamically create arguments with None (skip 'self')
        kwargs = {param: None for param in params if param != 'self'}
        
        # Call parent constructor with generated kwargs
        super().__init__(**kwargs)

    def show(self):
        print("Sequence of Trials Data:")
        print("- Use set_expansion_options() and expand_trials() to construct the sequence of trials dataset.")



class te_model_fitter(ABC):
    def __init__(self, save_path: str = ""):
        self.save_path = save_path

    # replace the below methods as necessary, depending on the subclasses.
    @abstractmethod
    def fit_outcome_model(self):
        """Method to fit outcome model - should be implemented by subclasses"""
        pass

    @abstractmethod
    def fit_weights_model(self):
        """Method to fit weights model - should be implemented by subclasses"""
        pass

    @abstractmethod
    def predict(self):
        """Method to make predictions - should be implemented by subclasses"""
        pass

class te_weights_fitted:
    """Class to store fitted weights model results."""
    def __init__(self, label: str, summary: dict, fitted: pd.Series):
        """
        Args:
            label (str): A short description of the model.
            summary (dict): Model summary with 'tidy', 'glance', and 'save_path' keys,
                           each containing a list of dicts (from DataFrame.to_dict('records')).
            fitted (pd.Series): Fitted values from the model.
        """
        self.label = label
        self.summary = summary
        self.fitted = fitted

    def show(self):
        """Display the fitted weights model details."""
        print(f"Model: {self.label}\n")
        for key in self.summary:
            # Convert list of dicts back to DataFrame for display
            df = pd.DataFrame(self.summary[key])
            if not df.empty:
                # Round numbers for cleaner output, similar to R's print.data.frame
                numeric_cols = df.select_dtypes(include=['float64']).columns
                df[numeric_cols] = df[numeric_cols].round(4)
                print(f"{key.capitalize()} summary:")
                display(df)  # Or use print(df.to_string(index=False)) for plain text
                print("")

    def __repr__(self):
        return f"TEWeightsFitted(label='{self.label}', summary_keys={list(self.summary.keys())})"
        

class te_stats_glm_logit(te_model_fitter):
    """A model fitter using logistic regression from statsmodels."""
    def __init__(self, save_path: str = ""):
        super().__init__(save_path=save_path)
        self.fitted_models = {}  # Store fitted models. change this line of code to be consistent with the docs

    def fit_weights_model(self, data: pd.DataFrame, formula: str, label: str) -> TEWeightsFitted:
        """
        Fit a logistic regression model for weights.

        Args:
            data (pd.DataFrame): Data to fit the model on.
            formula (str): Model formula (e.g., "treatment ~ age + x1").
            label (str): Identifier for the model.

        Returns:
            TEWeightsFitted: Object containing model results and fitted values.
        """
        # Fit the logistic regression model
        model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial())
        result = model.fit()

        # Save the model if save_path is specified
        save_file = ""
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            save_file = tempfile.mktemp(prefix="model_", dir=self.save_path, suffix=".pkl")
            result.save(save_file)

        # Prepare summary
        tidy_df = pd.DataFrame({
            "term": result.params.index,
            "estimate": result.params.values,
            "std.error": result.bse.values,
            "statistic": result.tvalues.values,
            "p.value": result.pvalues.values
        })
        glance_df = pd.DataFrame({
            "AIC": [result.aic],
            "BIC": [result.bic_llf],
            "logLik": [result.llf],
            "deviance": [result.deviance],
            "df.resid": [int(result.df_resid)]
        }, index=[0])
        summary = {
            "tidy": tidy_df.to_dict(orient="records"),
            "glance": glance_df.to_dict(orient="records"),
            "save_path": pd.DataFrame({"path": [save_file]}).to_dict(orient="records")
        }

        # Store fitted values and model
        fitted_values = result.fittedvalues
        self.fitted_models[formula] = result

        return te_weights_fitted(label=label, summary=summary, fitted=fitted_values)

    def fit_outcome_model(self): # change this method code
        """Not implemented for switch weights; included for abstract compliance."""
        raise NotImplementedError("fit_outcome_model not supported for TEStatsGLMLogit")

    def predict(self, data: pd.DataFrame, formula: str): # change this method code
        """Predict using the fitted model."""
        if formula not in self.fitted_models:
            raise ValueError(f"No fitted model for formula: {formula}")
        return self.fitted_models[formula].predict(data)

def stats_glm_logit(save_path: str = "") -> te_stats_glm_logit:
    """
    Factory function to create a TEStatsGLMLogit instance.

    Args:
        save_path (str, optional): Path to save fitted models. Defaults to "" (no saving).

    Returns:
        TEStatsGLMLogit: An instance of the model fitter.

    Raises:
        ValueError: If save_path is invalid and not empty.
    """
    if save_path:  # Not empty
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            raise ValueError(f"Directory for save_path '{save_path}' does not exist")
        # In R, assert_path_for_output allows overwrite; we'll assume overwriting is fine
    return te_stats_glm_logit(save_path=save_path)


class te_weights_spec:
    def __init__(self, numerator, denominator, pool_numerator, pool_denominator, model_fitter, fitted, data_subset_expr):
        self.numerator = numerator #"formula",
        self.denominator = denominator #"formula",
        self.pool_numerator = pool_numerator #"logical",
        self.pool_denominator = pool_denominator #"logical",
        self.model_fitter = model_fitter #"te_model_fitter",
        self.fitted = fitted #"list",
        self.data_subset_expr = data_subset_expr #"list"

        # Validate model_fitter
        if not isinstance(self, te_weights_unset) and not hasattr(self.model_fitter, "fit_weights_model"):
            raise ValueError(f"No fit_weights_model method found for object with model_fitter class {type(model_fitter).__name__}")

    def show(self):
        print(f" - Numerator formula: {self.numerator}")
        print(f" - Denominator formula: {self.denominator}")
        if self.pool_numerator:
            if self.pool_denominator:
                print(" - Numerator and denominator models are pooled across treatment arms.")
            else:
                print(" - Numerator model is pooled across treatment arms. Denominator model is not pooled.")
        print(f" - Model fitter type: {type(self.model_fitter).__name__}")
        if self.fitted:
            print(" - View weight model summaries with show_weight_models()")
        else:
            print(" - Weight models not fitted. Use calculate_weights()")

# class te_weights_switch(te_weights_spec):
# class te_weights_censoring(te_weights_spec):

class te_weights_unset(te_weights_spec):
    def __init__(self):
        # Get parameter names of parent class __init__
        params = inspect.signature(te_weights_spec.__init__).parameters
        
        # Dynamically create arguments with None (skip 'self')
        kwargs = {param: None for param in params if param != 'self'}
        
        # Call parent constructor with generated kwargs
        super().__init__(**kwargs)

    def show(self):
        print(" - No weight model specified")

class te_outcome_fitted:
    def __init__(self, model=None, summary=None):
        self.model = model if model is not None else []
        self.summary = summary if summary is not None else {}

    def show(self):
        if self.summary:
            print("Model Summary:\n")
            if "tidy" in self.summary:
                tidy_df = pd.DataFrame(self.summary["tidy"])
                print(tidy_df.round(2).to_string(index=False))  # Round to 2 decimals

            if "glance" in self.summary:
                print("\n")
                glance_df = pd.DataFrame(self.summary["glance"])
                print(glance_df.round(3).to_string(index=False))  # Round to 3 decimals
        else:
            print("Use fit_msm() to fit the outcome model")
            

class te_outcome_model:
    def __init__(self, formula, adjustment_vars, treatment_var, 
                 adjustment_terms, treatment_terms, followup_time_terms, trial_period_terms, 
                 stabilised_weights_terms, model_fitter, fitted = None):
        self.formula = formula # formula
        self.adjustment_vars = adjustment_vars # character
        self.treatment_var = treatment_var # character
        self.adjustment_terms = adjustment_terms # formula
        self.treatment_terms = treatment_terms # formula
        self.followup_time_terms = followup_time_terms # formula
        self.trial_period_terms = trial_period_terms # formula
        self.stabilised_weights_terms = stabilised_weights_terms # formula
        self.model_fitter = model_fitter # te_model_fitter
        self.fitted = fitted # te_outcome_fitted

    def show(self):
        print("- Formula:", self.formula)
        print("- Treatment variable:", self.treatment_var)
        print("- Adjustment variables:", ", ".join(self.adjustment_vars))
        print("- Model fitter type:", type(self.model_fitter).__name__)
        #print(" ")
        
        if self.fitted:
            self.fitted.show()  # Assuming `fitted` has a `show()` method     

class te_outcome_model_unset(te_outcome_model):
    def __init__(self):
        # Get parameter names of parent class __init__
        params = inspect.signature(te_outcome_model.__init__).parameters
        
        # Dynamically create arguments with None (skip 'self')
        kwargs = {param: None for param in params if param != 'self'}
        
        # Call parent constructor with generated kwargs
        super().__init__(**kwargs)
        
    def show(self):
        print(" - Outcome model not specified. Use set_outcome_model()")

class TrialSequence:
    def __init__(self, estimand, expansion=None, outcome_model=None):
        self.data = te_data_unset() # class te_data
        self.estimand = estimand
        self.expansion = te_expansion_unset() if expansion is None else expansion
        self.outcome_model = te_outcome_model_unset() if outcome_model is None else outcome_model
        self.censor_weights = te_weights_unset()  # Will be set later
        self.outcome_data = None

    def set_data(self, data, censor_at_switch, ID = "id", period = "period", treatment = "treatment", 
                 outcome = "outcome", eligible = "eligible"):
        # Ensure the provided data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Column validation
        required_cols = {ID, period, treatment, outcome, eligible}
        forbidden_cols = {"wt", "wtC", "weight", "dose", "assigned_treatment"}        

        if not required_cols.issubset(data.columns):
            missing_cols = required_cols - set(data.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if forbidden_cols & set(data.columns):
            raise ValueError(f"Data contains forbidden columns: {forbidden_cols & set(data.columns)}")
            
        # Check for duplicate column names
        if len(set([ID, period, treatment, outcome, eligible])) < 5:
            raise ValueError("Duplicate column names specified.")

        # Rename columns to standard names
        trial_data = data.rename(columns={
            ID: "id",
            period: "period",
            treatment: "treatment",
            outcome: "outcome",
            eligible: "eligible"
        })

        # Apply data manipulation (assume data_manipulation is a function defined elsewhere)
        trial_data = data_manipulation(trial_data, use_censor=censor_at_switch)
        self.data = te_data(data=trial_data)

        return self

    def set_switch_weight_model(self, numerator=None, denominator=None, model_fitter=None, eligible_wts_0=None, eligible_wts_1=None):
        """
        Set the switch weight model for a TrialSequence object.
    
        Args:
            object (TrialSequence): The trial sequence object to modify.
            numerator (str, optional): Formula for numerator (e.g., "~age"). Defaults to "~1".
            denominator (str, optional): Formula for denominator (e.g., "~age + x1"). Defaults to "~1".
            model_fitter (TEModelFitter): Object to fit the weight model (e.g., from stats_glm_logit).
            eligible_wts_0 (str, optional): Column name for eligibility weights (treatment 0).
            eligible_wts_1 (str, optional): Column name for eligibility weights (treatment 1).
    
        Returns:
            TrialSequence: Updated trial sequence object.
    
        Raises:
            ValueError: If data is unset, formulas are invalid, or required columns are missing.
        """
        # Check if data is set
        if isinstance(self.data, te_data_unset):
            raise ValueError("Please use set_data() to set up the data before setting switch weight models")
    
        # Check if switch weights are supported
        if isinstance(self, TrialSequenceITT):
            raise ValueError("Switching weights are not supported for intention-to-treat (ITT) analyses")
    
        # Access the pandas DataFrame
        df = self.data.data  # This is a pandas DataFrame
        cols = set(df.columns)
    
        # Handle eligible_wts_0 and eligible_wts_1 renaming
        if eligible_wts_0 is not None:
            if eligible_wts_0 not in cols:
                raise ValueError(f"Column '{eligible_wts_0}' not found in data")
            df = df.rename(columns={eligible_wts_0: "eligible_wts_0"})
        if eligible_wts_1 is not None:
            if eligible_wts_1 not in cols:
                raise ValueError(f"Column '{eligible_wts_1}' not found in data")
            df = df.rename(columns={eligible_wts_1: "eligible_wts_1"})
        self.data.data = df  # Update the DataFrame
    
        # Default formulas
        numerator = "~1" if numerator is None else numerator
        denominator = "~1" if denominator is None else denominator
    
        # Validate and update formulas
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise ValueError("numerator and denominator must be formula strings (e.g., '~age')")
        if "time_on_regime" in rhs_vars(numerator):
            raise ValueError("time_on_regime should not be used in numerator")
        numerator = update_formula(numerator)
        denominator = update_formula(denominator)
    
        # Ensure model_fitter is provided and valid
        if model_fitter is None:
            raise ValueError("model_fitter must be provided (e.g., stats_glm_logit())")
        if not isinstance(model_fitter, TEModelFitter):
            raise ValueError("model_fitter must be an instance of TEModelFitter")
    
        # Set switch_weights
        self.switch_weights = te_weights_spec(
            numerator=numerator,
            denominator=denominator,
            pool_numerator=False,
            pool_denominator=False,
            model_fitter=model_fitter,
            fitted=None,
            data_subset_expr=None
        )
    
        # Update outcome formula (placeholder)
        self = update_outcome_formula(self)
        return self

    def show(self):
        print("Trial Sequence Object")    
        print(f"Estimand: {self.estimand}")
        print("")
        print("Data:")
        self.data.show()
        print("")

        print("IPW for informative censoring:")
        self.censor_weights.show()

        if hasattr(self, "switch_weights"):
            print("\nIPW for treatment switch censoring:")
            self.switch_weights.show()
    
        print("")
    
        if not isinstance(self.data, te_data_unset):  # Assuming TEDataUnset is a class
            self.expansion.show()
            print("")
    
        print("Outcome model:")
        self.outcome_model.show()
        print("")
    
        if self.expansion.datastore.N > 0:
            self.outcome_data.show()

    # def __repr__(self):
    #     return (
    #         f"TrialSequence(Estimand={self.estimand}, "
    #         f"Data={'Loaded' if self.data is not None else 'None'}, "
    #         f"ID={self.id_col}, Period={self.period_col}, "
    #         f"Treatment={self.treatment_col}, Outcome={self.outcome_col}, Eligible={self.eligible_col}, "
    #         f"CensorAtSwitch={self.censor_at_switch})"
    #     )

class TrialSequenceITT(TrialSequence):
    def __init__(self, expansion=None, outcome_model=None):
        super().__init__(estimand="ITT", expansion=expansion, outcome_model=outcome_model)
        
    # def set_switch_weight_model(self, *args, **kwargs):
    #     raise ValueError("Switching weights are not supported for intention-to-treat (ITT) analyses")

    def set_data(self, data: pd.DataFrame, ID = "id", period = "period", treatment = "treatment", outcome = "outcome", eligible = "eligible"):
        super().set_data(data=data, censor_at_switch = False, ID = "id", period = "period", 
                         treatment = "treatment", outcome = "outcome", eligible = "eligible")


class TrialSequencePP(TrialSequence):
    def __init__(self, expansion=None, outcome_model=None):
        super().__init__(estimand="PP", expansion=expansion, outcome_model=outcome_model)
        self.switch_weights = te_weights_unset()  # Per-protocol also accounts for treatment switching

    def set_data(self, data: pd.DataFrame, ID = "id", period = "period", treatment = "treatment", outcome = "outcome", eligible = "eligible"):
        super().set_data(data=data, censor_at_switch = True, ID = "id", period = "period", 
                         treatment = "treatment", outcome = "outcome", eligible = "eligible")

class TrialSequenceAT(TrialSequence):
    def __init__(self, expansion=None, outcome_model=None):
        super().__init__(estimand="AT", expansion=expansion, outcome_model=outcome_model)
        self.switch_weights = te_weights_unset()  # As-treated requires switching weights

    def set_data(self, data: pd.DataFrame, ID = "id", period = "period", treatment = "treatment", outcome = "outcome", eligible = "eligible"):
        super().set_data(data=data, censor_at_switch = False, ID = "id", period = "period", 
                         treatment = "treatment", outcome = "outcome", eligible = "eligible")

