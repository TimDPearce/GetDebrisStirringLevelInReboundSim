
**- BUG NOTE: A previous version of this code contained a bug, which caused the collision velocity to be overestimated roughly 50% of the time. This bug was corrected on 26/01/2024. We thank Marcy Best and Antranik Sefilian for spotting it. -**

Python program to quantify the level of debris stirring in a Rebound n-body simulation, by Tyson Costa and Tim Pearce. The code was originally used in Costa, Pearce & Krivov (2023).

To access the program, first download and unpack the ZIP file (press the green 'Code' button on GitHub, then 'Download ZIP', then unzip the file on your computer).

The program requires Python 3. To analyse a Rebound simulation located at e.g. 'exampleSims/exampleSim1.bin', navigate to the directory where this code is saved, then in the command line type e.g.

python3 getDebrisStirringLevelInReboundSim.py exampleSims/exampleSim1.bin

***NOTE: this version requires that all bodies in the Rebound simulation were assigned unique hashes when the simulation was set up (see https://rebound.readthedocs.io/en/latest/ipython_examples/UniquelyIdentifyingParticlesWithHashes/ for details). This is because hashing safely identifies particles even if they get removed or their order rearranged during a simulation. It also requires that the Rebound simulation units are au, yr and mSun; this is set in the simulation by using sim.units = ('yr', 'AU', 'Msun'). Future versions of this code may allow unhashed bodies and arbitrary units, if there is sufficient demand to implement this.***

The program takes the .bin output file from a Rebound simulation, and quantifies the level of debris stirring at the end of the simulation. It does this by examining multiple simulation snapshots, and at each one, checking each debris particle against each other particle (with certain criteria) to determine if a collision is possible between these particles. If so, then the collision speeds at the orbit-intersection points are calculated and compared to the fragmentation speed for the particles (of a chosen size, assuming a basalt composition by default); if the collision speed is greater than the fragmentation speed, the particles would undergo a destructive collision and are classified as stirred. If particles are scattered (i.e. their semimajor axes change by a significant amount), then they are classified as scattered. All remaining particles are classified as unstirred. The code outputs the total numbers of stirred, scattered and unstirred debris bodies, and also bins them by initial semimajor axis. The code can display the results as a plot, save the plot, and also save the data to CSV files.

You can change specific analysis values in the "User Inputs" section; no other part of the code should need to be changed. The default settings are those identified as best in Costa, Pearce & Krivov (2023). The fragmentation-speed prescription is defined in the function GetFragmentationSpeedForBasalt(); replace this if you want a different material or prescription.

Feel free to use this program to quantify stirring for your own Rebound simulations. If the results go into a publication, then please cite Costa, Pearce & Krivov (2023). Please let the authors know if you find any bugs or have any requests. In particular, different users have very different conventions for setting up Rebound simulations; if yours aren't compatible with this code but you feel they should be, then we would be very interested to hear from you.
