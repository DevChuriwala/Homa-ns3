# Print hi
echo "Running ./waf configure"
sleep 0.3
./waf configure

echo "\nRunning ./waf"
sleep 0.3
./waf

echo "\nRunning simple-test"
sleep 0.3
./waf --run scratch/HomaL4Protocol-simple-topology