import pandas as pd
from IPython.display import display
# import statsmodels.api as sm
# import os
# import tempfile
# import patsy
# from typing import Optional
# import statsmodels.formula.api as smf
from abc import ABC, abstractmethod
import inspect
import numpy as np

# Trial sequence class and function definitions
def trial_sequence(estimand, **kwargs):
    estimand_classes = {
        "ITT": TrialSequenceITT,
        "PP": TrialSequencePP,
        "AT": TrialSequenceAT,
    }
    
    if estimand not in estimand_classes:
        raise ValueError(f"Invalid estimand: {estimand}. Choose from ITT, PP, AT.")
    
    return estimand_classes[estimand](**kwargs)

def te_outcome_data(data: pd.DataFrame, p_control: float = None, subset_condition: str = None):
    # Validate input data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    # Required columns check
    missing_columns = TEOutcomeData.REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Compute required values
    n_rows = len(data)
    if n_rows == 0:
        print("Warning: Outcome data has 0 rows")

    n_ids = data["id"].nunique()
    periods = sorted(data["trial_period"].unique())

    # Default values if None
    subset_condition = subset_condition if subset_condition is not None else ""
    p_control = p_control if p_control is not None else 0.0

    # Return an instance of TEOutcomeData
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
    n = len(sw_data)
    
    started0 = sw_data["started0"].copy()
    started1 = sw_data["started1"].copy()
    stop0 = sw_data["stop0"].copy()
    stop1 = sw_data["stop1"].copy()
    eligible0_sw = sw_data["eligible0_sw"].copy()
    eligible1_sw = sw_data["eligible1_sw"].copy()
    delete = sw_data["delete"].copy()
    
    t_first = sw_data["first"].astype(bool)
    t_eligible = sw_data["eligible"].astype(bool)
    t_treatment = sw_data["treatment"].astype(int)
    t_switch = sw_data["switch"].astype(int)
    
    started0_ = started1_ = stop0_ = stop1_ = eligible0_sw_ = eligible1_sw_ = 0
    
    for i in range(n):
        if t_first[i]:
            started0_ = started1_ = stop0_ = stop1_ = eligible0_sw_ = eligible1_sw_ = 0
        
        if stop0_ == 1 or stop1_ == 1:
            started0_ = started1_ = stop0_ = stop1_ = eligible0_sw_ = eligible1_sw_ = 0
        
        if started0_ == 0 and started1_ == 0 and t_eligible[i]:
            if t_treatment[i] == 0:
                started0_ = 1
            elif t_treatment[i] == 1:
                started1_ = 1
        
        if started0_ == 1 and stop0_ == 0:
            eligible0_sw_ = 1
            eligible1_sw_ = 0
        elif started1_ == 1 and stop1_ == 0:
            eligible0_sw_ = 0
            eligible1_sw_ = 1
        else:
            eligible0_sw_ = eligible1_sw_ = 0
        
        if t_switch[i] == 1:
            if t_eligible[i]:
                if t_treatment[i] == 1:
                    started1_ = 1
                    stop1_ = 0
                    started0_ = 0
                    stop0_ = 0
                    eligible1_sw_ = 1
                elif t_treatment[i] == 0:
                    started0_ = 1
                    stop0_ = 0
                    started1_ = 0
                    stop1_ = 0
                    eligible0_sw_ = 1
            else:
                stop0_ = started0_
                stop1_ = started1_
        
        if eligible0_sw_ == 0 and eligible1_sw_ == 0:
            delete[i] = True
        else:
            started0[i] = started0_
            started1[i] = started1_
            stop0[i] = stop0_
            stop1[i] = stop1_
            eligible1_sw[i] = eligible1_sw_
            eligible0_sw[i] = eligible0_sw_
            delete[i] = False
    
    sw_data["started0"] = started0
    sw_data["started1"] = started1
    sw_data["stop0"] = stop0
    sw_data["stop1"] = stop1
    sw_data["eligible0_sw"] = eligible0_sw
    sw_data["eligible1_sw"] = eligible1_sw
    sw_data["delete"] = delete
    
    return sw_data

class TEOutcomeData: # this is the te_outcome_data class in the R TrialEmulation package docs
    REQUIRED_COLUMNS = {"id", "trial_period", "followup_time", "outcome", "weight"}
    
    def __init__(self, data: pd.DataFrame, n_rows: int, n_ids: int, periods: int, p_control: float, subset_condition: str):
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
            print(f"N: {self.n_rows} observations from {self.n_ids} patients in {len(self.periods)} trial periods")
            print(f"Periods: {self.periods}")
            if self.subset_condition:
                print(f"Subset condition: {self.subset_condition}")
            if self.p_control is not None:
                print(f"Sampling control observations with probability: {self.p_control}")
            
            # Print a sample of the data, similar to R's print(data, nrows=4, topn=2)
            display(self.data.head(4))  # Show first 4 rows
        
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

    #def set_switch_weight_model():

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

