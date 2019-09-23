from clean_data import clean_class
import pandas as pd


for x in range(1, 5):
  globals()['data%s' % x] = pd.read_csv('test' + str(x) + '.csv', encoding ="latin1")
# Concatanete them
df = pd.concat([data1, data2 ,data3, data4])
df = df.sort_values(['employee_name' , 'timestamp', 'camera_id'])
df['unified_id'] = df['camera_id']
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

df['unified_id'] = df['unified_id'].replace(2, 1)
df['unified_id'] = df['unified_id'].replace(4, 3)
####df['unified_id'] = df['unified_id'].replace(6, 5)


####temp = DataFrame(entry)
####df = df.append(temp)
####df.to_csv('final.csv', index = False)



entries = {'employee_name': [],
      'arrival_time': [],
      'depart_time':[],
      'on_working':[],
      'off_working':[],
      'fun_area_time':[]
      }
	  
entries_df = pd.DataFrame(entries)

#df.to_csv('final.csv', index = False)

location = 0
for name in df.employee_name.unique():

	df_emp = df[df['employee_name'] == name]
	df_emp = clean_class(df_emp)
	df_emp = df_emp.groupby((df_emp['unified_id'] != df_emp['unified_id'].shift()).cumsum().values).first()

	first_seen = -1
	last_seen = -1

	employee_data = {'entrance':0.0,
          'exit': 0.0,
          'on_working':0.0,
          'off_working':0.0
          }

	for index, row in df_emp.iterrows():
   		
   		if(row['unified_id'] == 1):
   			if(first_seen == -1):            
   				first_seen = row['timestamp']

   			if(employee_data['entrance'] == 0):
   				employee_data['entrance'] = row['timestamp']
   				if(employee_data['exit'] != 0):
   					employee_data['off_working'] = employee_data['off_working'] + (employee_data['entrance'] - employee_data['exit'])
   					employee_data['exit'] = 0
   			else:
   				continue



   		elif(row['unified_id'] == 3):
   			if(employee_data['entrance'] != 0):
   				employee_data['exit'] = row['timestamp']
   				employee_data['on_working'] = employee_data['on_working'] + (employee_data['exit'] - employee_data['entrance'])
   				last_seen = employee_data['exit']  
   				employee_data['entrance'] = 0
#########################################
	entry = [name, first_seen, last_seen, employee_data['on_working'], employee_data['off_working'], 0]
	entries_df.loc[location] = entry
	location = location + 1

entries_df.to_csv('result.csv', index = False)
##########################################