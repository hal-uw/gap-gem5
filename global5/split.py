# Open the file and read its contents
with open('simout.txt', 'r') as f:
    lines = f.read()

# Split content by the '[GLOBAL]' keyword
sections = lines.split('[GLOBAL]')

# Loop through each section and save to a separate file
for i, section in enumerate(sections[1:], start=1):
    with open('global_section_{}.txt'.format(i), 'w') as out_file:
        out_file.write('[GLOBAL]{}'.format(section))

