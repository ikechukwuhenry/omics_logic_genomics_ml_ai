print("hello from r", )
data <- read.table('illumina_genes.txt', header = TRUE,  sep = '	')

print(summary(data))
x <- data[2]
y <- data[3]
# print(x)

x <- data[1:1000, 2]
y <- data[1:1000, 3]


plot(x, type = "l", col = "blue")
lines(y, type = "l", col = "red")

library(readxl)
# Basic usage
# cancer_data <- read_excel("TCGA Gastric TPM Corrected.xlsx")
# print(summary(cancer_data))

# With additional options
cancer_data <- read_excel("TCGA Gastric TPM Corrected.xlsx",
                          sheet = 1,           # or sheet name "Sheet1"
                          col_names = TRUE,    # first row contains column names
                          skip = 1,            # skip first n rows
                          n_max = Inf)         # maximum number of rows to read

# print(summary(cancer_data))

# For a specific row (e.g., row 1)
unique_values <- unique(cancer_data[1, ])
print((unique_values))

# transpose the data the columns (samples) becomes
# the rows and the genes becomes the column
transpose_cancer_data <- t(cancer_data)

# path = "/Users/ikechukwumichael/Desktop/r_codes/"
write.table(transpose_cancer_data,
            file = "/Users/ikechukwumichael/Desktop/r_codes/TCGA Gastric TPM-complete.csv",
            quote = FALSE, col.names = FALSE, sep = ",")
