Put your sales file here as superstore.csv (or set the SALES_CSV environment variable to its path)
and the app loads it on start. You can also upload a CSV/XLSX from the page itself.

Required columns: Order Date, Sales. Everything else is optional; charts that need a missing
column say so instead of failing. Column names are matched loosely ("Sub-Category", "Sub Category"
and "sub_category" are all fine).

Data files are git-ignored so you don't publish sales data by accident.
