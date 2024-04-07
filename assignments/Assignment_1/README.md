## README

### Operations on Dataframes

#### Reading Data
- Data can be read from both CSV and XLS files using `pd.read_csv()` and `pd.read_excel()` functions, respectively.

#### Indexing
- Indexing can be performed in multiple ways:
  - By column name: `df['column_name']`
  - By row index: `df.loc['row_name']`
  - By row and column index: `df.iloc[row_index, column_index]`

#### Sorting
- Dataframes can be sorted using the `sort_values()` function, specifying the column by which to sort.

#### Data Types
- To check data types of columns: `df.info()`
- To convert variable data types: `pd.to_numeric()`

#### Descriptive Statistics
- Descriptive statistics of numerical columns: `df.describe()`

#### Missing Values
- To identify missing values: `df.isnull()`
- To fill missing values: `df.fillna(value)`

### Conclusion
Various operations can be performed on dataframes in pandas, including indexing, sorting, handling missing values, and converting data types. These operations are essential for data preprocessing and analysis.
