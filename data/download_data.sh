set -e

# Get data from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

# Download data if not already downloaded:
if [ ! -f ./kddcup99.html ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
fi
if [ ! -f ./task.html ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/task.html
fi
if [ ! -f ./kddcup.names ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
fi
if [ ! -f ./kddcup.data.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
fi
if [ ! -f ./kddcup.data_10_percent.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
fi
if [ ! -f ./kddcup.newtestdata_10_percent_unlabeled.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.newtestdata_10_percent_unlabeled.gz
fi
if [ ! -f ./kddcup.testdata.unlabeled.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled.gz
fi
if [ ! -f ./kddcup.testdata.unlabeled_10_percent.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz
fi
if [ ! -f ./corrected.gz ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
fi
if [ ! -f ./training_attack_types ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
fi
if [ ! -f ./typo-correction.txt ]; then
    wget http://kdd.ics.uci.edu/databases/kddcup99/typo-correction.txt
fi

# Unzip data:
if [ ! -f ./kddcup.data ]; then
    gunzip kddcup.data.gz
fi
if [ ! -f ./kddcup.data_10_percent ]; then
    gunzip kddcup.data_10_percent.gz
fi
if [ ! -f ./kddcup.newtestdata_10_percent_unlabeled ]; then
    gunzip kddcup.newtestdata_10_percent_unlabeled.gz
fi
if [ ! -f ./kddcup.testdata.unlabeled ]; then
    gunzip kddcup.testdata.unlabeled.gz
fi
if [ ! -f ./kddcup.testdata.unlabeled_10_percent ]; then
    gunzip kddcup.testdata.unlabeled_10_percent.gz
fi
if [ ! -f ./corrected ]; then
    gunzip corrected.gz
fi
