import pandas as pd

def clean_class(df_emp, confidence_interval = 5):

	doubted_entry = None
	confirmed_entry = None
	confidence_level = confidence_interval
	location = 0
	counter = 0
	first_entry = True
	clean_df = pd.DataFrame(columns= df_emp.columns.tolist())

	for index, row in df_emp.iterrows():

		if(first_entry == True):
			doubted_entry = row
			first_entry = False
			counter = counter + 1
			continue


		if(row['unified_id'] == doubted_entry['unified_id']):
			confidence_level = confidence_level - 1

			if((counter == len(df_emp.index) - 1) and confidence_level <= 0):
				confirmed_entry = doubted_entry
				clean_df.loc[location] = confirmed_entry.tolist()

		elif(row['unified_id'] != doubted_entry['unified_id'] and confidence_level <= 0):
			confirmed_entry = doubted_entry
			clean_df.loc[location] = confirmed_entry.tolist()

			location = location + 1
			doubted_entry = row
			confidence_level = confidence_interval
		else:
			doubted_entry = row
			confidence_level = confidence_interval

		counter = counter + 1

	return clean_df

