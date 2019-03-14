def create_file(input, output):
	with open(input, 'r') as fr:
		with open(output, 'w') as fw:
			for line in fr:
				fw.write(line[4:-6])
				fw.write('\n')


create_file('/disk/scratch/mpatsis/eddie/data/java_training_slp_pre', 
	'/disk/scratch/mpatsis/eddie/data/elmo/java_training_slp_pre')
create_file('/disk/scratch/mpatsis/eddie/data/java_validation_slp_pre', 
        '/disk/scratch/mpatsis/eddie/data/elmo/java_validation_slp_pre')
create_file('/disk/scratch/mpatsis/eddie/data/java_test_slp_pre', 
        '/disk/scratch/mpatsis/eddie/data/elmo/java_test_slp_pre')
create_file('/disk/scratch/mpatsis/eddie/data/java_training_slp_huge_pre', 
        '/disk/scratch/mpatsis/eddie/data/elmo/java_training_slp_huge_pre')
