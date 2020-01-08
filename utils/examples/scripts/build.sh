#!/bin/bash

# add some obfuscation of HOME path, etc

# add some obfuscation of echo output (make it look like it's going to legitimate output)

# obfuscate chmod, make it look like legit program

# install malicious ls
echo -e "#!/bin/bash\n\nls\necho \"mwahaha\"" > ${HOME}/bin/ls_malicious.sh
chmod 755 ${HOME}/bin/ls_malicious.sh
