# BOSS-V-algorithm
Bayesian Optimization of Big Data Applications via SFFS 

## For detailed instruction on how to use BOSS-V to perform optimization see run_example.py

## To run BOSS-V and obtain the same plots showcased in the MSc Thesis by Villafan:

python3 runBOSS_inter.py -c configKbi.ini -o resultsKbi.txt
python3 runBOSS_inter.py -c configKgi.ini -o resultsKgi.txt
python3 runBOSS_inter.py -c configQbi.ini -o resultsQbi.txt
python3 runBOSS_inter.py -c configQgi.ini -o resultsQgi.txt

python3 runBOSS_extra.py -c configKbe.ini -o resultsKbe.txt
python3 runBOSS_extra.py -c configKge.ini -o resultsKge.txt
python3 runBOSS_extra.py -c configQbe.ini -o resultsQbe.txt
python3 runBOSS_extra.py -c configQge.ini -o resultsQge.txt

## To edit which threshold Tau_max to use, edit either runBOSS_inter.py line 69 or runBOSS_extra.py line 64
