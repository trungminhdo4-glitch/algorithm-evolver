"""
Generic CSV Problem: Load datasets for scientific discovery from CSV files.
"""
import os
import pandas as pd
import numpy as np
import re
from core.physics import Dimension

class CsvProblem:
    """
    A generic problem class that loads data from a CSV file.
    Infers input/output structure and attempts to parse units.
    """
    
    def __init__(self, file_path, target_column, drop_columns=None, units_dict=None):
        """
        Args:
            file_path (str): Path to CSV data.
            target_column (str): Name of the column to predict (y).
            drop_columns (list): Columns to ignore.
            units_dict (dict): Manual dimension overrides {column_name: Dimension(...)}.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.units_dict = units_dict or {}
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        self.df = pd.read_csv(file_path)
        
        # Drop columns
        if self.drop_columns:
            self.df = self.df.drop(columns=self.drop_columns)
            
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {self.df.columns}")
            
        # Sanitize column names for internal symbolic use
        self.original_names = list(self.df.columns)
        self.sanitized_names = [self._sanitize_name(col) for col in self.original_names]
        self.name_map = dict(zip(self.original_names, self.sanitized_names))
        self.inverse_name_map = dict(zip(self.sanitized_names, self.original_names))
        
        # Separate inputs and target (using original names for DF indexing, sanitized for GP)
        self.original_input_names = [col for col in self.df.columns if col != target_column]
        self.input_names = [self.name_map[col] for col in self.original_input_names]
        self.sanitized_target = self.name_map[target_column]
        
        self.name = os.path.basename(file_path).replace('.csv', '')
        
        # Prepare training data
        self.train_data = []
        for _, row in self.df.iterrows():
            inputs = tuple(row[self.original_input_names].values)
            target = row[self.target_column]
            self.train_data.append((inputs, target))
            
        # Infer Dimensional Units (using original names for units_dict check)
        self.pset_units = {}
        for col in self.original_names:
            sanitized = self.name_map[col]
            if col in self.units_dict:
                self.pset_units[sanitized] = self.units_dict[col]
            else:
                self.pset_units[sanitized] = self._parse_dimension(col)
                
        self.target_unit = self.pset_units[self.sanitized_target]
        self.pset_input_units = {}
        
        # Map ARG0, ARG1... to units for DimensionalChecker
        for i, col in enumerate(self.original_input_names):
            sanitized = self.name_map[col]
            unit = self.pset_units[sanitized]
            self.pset_input_units[f"ARG{i}"] = unit
            self.pset_input_units[sanitized] = unit # Also keep sanitized for other lookups
        
        # Add common ephemeral names (allow override from units_dict)
        self.pset_input_units['rand_float'] = self.units_dict.get('rand_float', Dimension(0, 0, 0, 0, 0))
        self.pset_input_units['rand_dimensional'] = self.units_dict.get('rand_dimensional', Dimension(0, 0, 0, 0, 0))
        
    def _sanitize_name(self, name):
        """Removes spaces, brackets, and other non-alphanumeric chars for Python safety."""
        # Replace non-word chars with underscores
        s = re.sub(r'[^\w]', '_', name)
        # Remove consecutive underscores
        s = re.sub(r'_+', '_', s)
        # Remove leading/trailing underscores
        return s.strip('_')

    def _parse_dimension(self, column_name):
        """
        Attempts to parse dimensions from column names like 'Mass [kg]' or 'Force [N]'.
        Defaults to Dimensionless.
        """
        # Basic mapping for known units/keywords
        # regex search for content in brackets
        match = re.search(r'\[(.*?)\]', column_name)
        if not match:
            # Fallback for common names
            lower = column_name.lower()
            if 'mass' in lower or lower == 'm1' or lower == 'm2': return Dimension(1, 0, 0, 0, 0)
            if 'dist' in lower or 'length' in lower or lower == 'r' or lower == 'x': return Dimension(0, 1, 0, 0, 0)
            if 'time' in lower or lower == 't': return Dimension(0, 0, 1, 0, 0)
            if 'force' in lower or lower == 'f': return Dimension(1, 1, -2, 0, 0)
            return Dimension(0, 0, 0, 0, 0)
            
        unit = match.group(1).lower()
        # Fixed logic: 'kg' is mass, 'm' is length
        if unit in ['kg', 'g']: return Dimension(1, 0, 0, 0, 0)
        if unit in ['m', 'km', 'cm']: return Dimension(0, 1, 0, 0, 0)
        if unit in ['s', 'sec', 'h']: return Dimension(0, 0, 1, 0, 0)
        if unit in ['n', 'newton']: return Dimension(1, 1, -2, 0, 0)
        
        return Dimension(0, 0, 0, 0, 0)

    def create_primitive_set(self, style="power"):
        """Creates a primitive set based on the CSV structure."""
        from core.primitives import create_power_law_primitive_set, create_typed_primitive_set
        
        num_inputs = len(self.input_names)
        input_types = [float] * num_inputs
        
        if style == "power":
            pset = create_power_law_primitive_set(input_types, float, name=f"CSV_{self.name.upper()}")
        else:
            pset = create_typed_primitive_set(input_types, float, name=f"CSV_{self.name.upper()}")
            
        # Map argument names to CSV columns
        kwargs = {f"ARG{i}": name for i, name in enumerate(self.input_names)}
        pset.renameArguments(**kwargs)
        
        return pset

    def evaluate(self, individual, func):
        """Standard evaluation logic."""
        from core.physics import DimensionalChecker
        from core.simplification import ProgramSimplifier
        import sympy as sp
        
        if not hasattr(self, '_eval_count'):
            self._eval_count = 0
            self._consistent_count = 0
            self._target_dim_count = 0
            
        self._eval_count += 1
        
        # 1. Dimensional Check
        checker = DimensionalChecker(self.pset_input_units)
        final_unit, consistent = checker.check_tree(individual)
        
        if consistent:
            self._consistent_count += 1
            if final_unit == self.target_unit:
                self._target_dim_count += 1
        
        if self._eval_count % 1000 == 0:
            print(f"   [Eval Stats] Total: {self._eval_count}, Consistent: {self._consistent_count}, Target Dim: {self._target_dim_count}")
        
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1e6 # Still severe for nonsense
        elif final_unit != self.target_unit:
            dim_penalty = 1000.0 # Softer penalty for wrong dimension
            
        # 2. Accuracy
        # func is pre-compiled
        mse = 0
        for inputs, target in self.train_data:
            try:
                output = func(*inputs)
                mse += (output - target)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                mse += 1e12
        
        mse /= len(self.train_data)
        accuracy_score = mse * dim_penalty
        
        # 3. Complexity
        try:
            simplifier = ProgramSimplifier()
            # We need a fresh pset naming context for simplification
            pset = self.create_primitive_set(style="power")
            simplified_str = simplifier.simplify_individual(individual, pset)
            
            if "Error" in simplified_str:
                complexity = len(individual)
            else:
                expr = sp.sympify(simplified_str)
                complexity = sp.count_ops(expr) + len(expr.free_symbols) + 1
        except Exception:
            complexity = len(individual)
            
        return accuracy_score, float(complexity)


class CsvLoaderProblem:
    """
    A generic class for loading CSV data and assigning physical units.
    Stricly follow the user's protocol for 'Universal CSV Loader'.
    """
    def __init__(self, file_path, target_column, unit_map, transformation=None):
        """
        Args:
            file_path (str): Path to CSV.
            target_column (str): Variable to predict.
            unit_map (dict): Mapping column names to [M, L, T, N, Theta].
            transformation (lambda): Optional function to transform target column.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.unit_dict = {k: Dimension(*v) if isinstance(v, (list, tuple)) else v 
                          for k, v in unit_map.items()}
        self.transformation = transformation
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
            
        self.df = pd.read_csv(file_path)
        
        # Apply transformation if present
        if self.transformation:
            self.df[target_column] = self.df[target_column].apply(self.transformation)
            
        self.input_columns = [col for col in self.df.columns if col != target_column]
        
        # Prepare training data (list of tuples for fast iteration)
        self.train_data = []
        for _, row in self.df.iterrows():
            inputs = tuple(row[self.input_columns].values)
            target = row[self.target_column]
            self.train_data.append((inputs, target))
            
        # Units for DimensionalChecker
        self.pset_units = {}
        for i, col in enumerate(self.input_columns):
            # Map default ARG0, ARG1... names to units
            self.pset_units[f"ARG{i}"] = self.unit_dict.get(col, Dimension(0,0,0,0,0))
            
        self.target_unit = self.unit_dict.get(target_column, Dimension(0,0,0,0,0))

    def evaluate(self, individual, compiled_func=None):
        """
        Calculates fitness based on MSE and physical consistency.
        """
        from core.physics import DimensionalChecker
        
        # 1. Dimensional Check
        checker = DimensionalChecker(self.pset_units)
        final_unit, consistent = checker.check_tree(individual)
        
        # Penalty calculation
        penalty = 1.0
        if not consistent:
            penalty = 1e8 # Extreme penalty for physical nonsense
        elif final_unit != self.target_unit:
            penalty = 1e4 # High penalty for wrong dimension
            
        # 2. Performance (MSE)
        # compiled_func can be passed in for multiprocessing efficiency
        func = compiled_func
        mse = 0
        try:
            for inputs, target in self.train_data:
                res = func(*inputs)
                mse += (res - target)**2
            mse /= len(self.train_data)
        except (ZeroDivisionError, OverflowError, ValueError):
            mse = 1e12
            
        # 3. Complexity
        complexity = len(individual)
        
        # Weighted score (Fitness is (WeightedMSE, Complexity))
        return mse * penalty, float(complexity)

    def __str__(self):
        return f"CsvLoaderProblem({self.target_column} from {os.path.basename(self.file_path)})"
