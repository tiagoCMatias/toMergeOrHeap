array_file = 'GeneratedData/output_30.csv'

data = pd.read_csv(array_file, sep=",", engine='python', names=['ID', 'length', 'array'])

for index, row in data.iterrows():
    new_array = row['array'].replace("]", "").replace("[", "")

test_array = [int(s) for s in new_array.split('  ')]

maior = test_array[0]
menor = test_array[0]
same_number = -1
last_number = test_array[0]
for item in test_array:
    if item > maior:
        maior = item
    if item < menor:
        menor = item
    if last_number == item:
        same_number += 1
    last_number = item

# plt=data.array.value_counts().plot()
repeated_values = data.array.value_counts().mean()

print("Maior: ", maior)
print("Menor: ", menor)

print("Mean Value: ", repeated_values)
print("Tamanho: ", row['length'])

index = data['ID'].iloc[0]