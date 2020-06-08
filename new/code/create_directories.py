import os


root_path = '../ZSH Carto/'
target_path = '../zsh_results/'
for person in os.listdir(root_path):
    if person == '.DS_Store':
        continue
    else:
        try:
            person_path = target_path + person
            os.mkdir(person_path)
        except OSError:
            print('Creation of the directory {} failed'.format(person_path))
        else:
            print('Successfully created the directory {}'.format(person_path))

