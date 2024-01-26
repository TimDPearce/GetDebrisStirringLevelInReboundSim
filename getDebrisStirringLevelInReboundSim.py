'''Python program to quantify the level of debris stirring in a Rebound
n-body simulation, by Tyson Costa and Tim Pearce. The code was
originally used in Costa, Pearce & Krivov (2023).  

The program requires Python 3. To analyse a Rebound simulation located at
e.g. 'exampleSims/exampleSim1.bin', navigate to the directory where this
code is saved, then in the command line type e.g.

python3 getDebrisStirringLevelInReboundSim.py exampleSims/exampleSim1.bin

***NOTE: this version requires that all bodies in the Rebound simulation
were assigned unique hashes when the simulation was set up (see
https://rebound.readthedocs.io/en/latest/ipython_examples/UniquelyIdentifyingParticlesWithHashes/
for details). This is because hashing safely identifies particles even
if they get removed or their order rearranged during a simulation. It
also requires that the Rebound simulation units are au, yr and mSun; this
is set in the simulation by using sim.units = ('yr', 'AU', 'Msun').
Future versions of this code may allow unhashed bodies and arbitrary 
units, if there is sufficient demand to implement this.***

The program takes the .bin output file from a Rebound simulation, and
quantifies the level of debris stirring at the end of the simulation. It
does this by examining multiple simulation snapshots, and at each one,
checking each debris particle against each other particle (with certain
criteria) to determine if a collision is possible between these
particles. If so, then the collision speeds at the orbit-intersection
points are calculated and compared to the fragmentation speed for the
particles (of a chosen size, assuming a basalt composition by default);
if the collision speed is greater than the fragmentation speed, the
particles would undergo a destructive collision and are classified as
stirred. If particles are scattered (i.e. their semimajor axes change by
a significant amount), then they are classified as scattered. All
remaining particles are classified as unstirred. The code outputs the
total numbers of stirred, scattered and unstirred debris bodies, and also
bins them by initial semimajor axis. The code can display the results as
a plot, save the plot, and also save the data to CSV files.

You can change specific analysis values in the "User Inputs" section
below; no other part of the code should need to be changed. The default 
settings are those identified as best in Costa, Pearce & Krivov (2023).
The fragmentation-speed prescription is defined in the function
GetFragmentationSpeedForBasalt(); replace this if you want a different
material or prescription.


***
BUG NOTE: A previous version of this code contained a bug, which
caused the collision velocity to be overestimated roughly 50% of the
time. This bug was corrected on 26/1/24. We thank Marcy Best and Antranik
Sefilian for spotting it.
***


Feel free to use this program to quantify stirring for your own Rebound
simulations. If the results go into a publication, then please cite 
Costa, Pearce & Krivov (2023). Please let the authors know if you find
any bugs or have any requests. In particular, different users have very
different conventions for setting up Rebound simulations; if yours aren't
compatible with this code but you feel they should be, then we would be
very interested to hear from you.
'''

############################### LIBRARIES ###############################
import sys
import os
import rebound
from rebound import hash as reboundHash
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import time as systime

############################### USER INPUTS ###############################
'''
Parameters for the code that the user can define themselves to alter the
functionality as they see fit, also defined with their units. No other 
part of the code should need to be changed outside of these parameters.
'''

# Size of particle considered for the stirring calculations. If the
# intersection speed of two n-body objects is sufficient that bodies of
# this size would undergo destructive collisions, then those bodies are
# considered stirred
stirredParticleSize_m = 0.01

# Number of semimajor axis bins across disk
numberSemimajorAxisBins = 10

# Interval over which to check snapshots (1 = check every snapshot,
# 2 = every other snapshot etc.). Lower values are more accurate but
# slower. The final snapshot is always checked.
snapshotInterval = 20

# Only pairs of debris particles on 'nearby' orbits are checked to see if
# they would collide destructively. This is done to prevent certain
# misleading situations; for example, preventing unexcited debris at the
# outer edge of a disk from being registered as stirred if bodies from
# the inner edge are scattered by planets onto highly eccentric orbits.
# There are three parameters used to define 'nearby' orbits:
#	- particleNearRange_percent is how close in *current* semimajor axis
#		two particles should be (in percent)
#	- particleInitialRange_percent is how close in *initial* semimajor
#		axis two particles should be (in percent)
#   - inclinationRange_deg is how close in inclination the particle 
#		orbits should be (in degrees)
particleNearRange_percent = 1.
particleInitialRange_percent = 1.
inclinationRange_deg = 1.

# A particle is classified as 'scattered' if its semimajor axis ever
# changes by a significant amount from its initial value. Set that amount
# here (in percent)
minSemimajorAxisChangeToBeScattered_percent = 20.

# Set how debris and non-debris bodies are to be identified from the
# Rebound simulation. If the following is set to True, then all 
# non-active bodies are classed as debris (to be used if debris was
# assigned as non-active during the Rebound- simulation setup). If False,
# then debris will instead be assumed to be any body smaller than a given
# mass (defined below).
shouldDebrisBeAssignedAsAllNonActiveBodies = True

# If not classifying debris as all non-active bodies (i.e. if the above 
# is set to shouldDebrisBeAssignedAsAllNonActiveBodies = False), then
# debris will be classified as any body below some mass instead. This
# critical mass is set here, in Earth masses. This is not used in any 
# calculations, but used to decide which simulation objects are debris
# and which are planets, stars etc.; when loading the simulation, any 
# masses smaller than this at the initial snapshot are classed as debris,
# whilst any larger ones are assumed to be other objects.
massDivideBetweenDebrisAndNonDebris_mEarth = 10.

# Should a plot be displayed
shouldPlotBeMade = True

# Should a plot be saved, and if so, its save path
shouldPlotBeSaved = False
outputPlotFilePath = 'outputs/images/testSimPlot.png'

# Should output data be saved as .csv files, and if so, their save paths
shouldDataBeSaved = False
outputCSVOutcomesByA0BinFilePath = 'outputs/data/stirringCodeOutcomesBySemimajorAxisBin.csv'
outputCSVTotalOutcomesFilePath = 'outputs/data/stirringCodeOutcomes.csv'

################################ CONSTANTS ##############################
# Solar mass in Earth masses
mSun_mEarth = 333000.

######################### BEGINNING OF FUNCTIONS ########################
def GetFragmentationSpeedForBasalt(stirredParticleSize_m):
	'''Returns the fragmentation speed of a body of a given radius
	assuming it is made of Basalt and that collisions occur between
	equal-sized bodies (e.g. Krivov et al. 2005). This is the minimum
	collision speed resulting in destructive collisions.

	INPUTS:
		- stirredParticleSize_m: Radius of the body

	OUTPUTS:
		- fragmentationSpeed_mPerS: fragmentation speed
	'''

	fragmentationSpeed_mPerS = 17.5 * ((stirredParticleSize_m)**(-0.36) + (stirredParticleSize_m/1000)**(1.4))**(2/3)

	return fragmentationSpeed_mPerS

#------------------------------------------------------------------------
def GetIfBodyIsDebris(bodyAtInitialSnapshot, bodyIndexAtInitialSnapshot, initialNumberOfActiveBodies):
	'''Determine whether the body, supplied at the initial simulation 
	snapshot, should be classed as debris or not
	
	INPUTS:
		- bodyAtInitialSnapshot: The body to be classified, at the 
			initial snapshot
		- bodyIndexAtInitialSnapshot: Index of the body at the initial
			snapshot
		- initialNumberOfActiveBodies: Number of bodies set to active at
			the initial snapshot

	OUTPUTS:
		- isBodyDebris: boolean for whether body is debris or not
	'''

	# Default is that body is debris. Will be checked and potentially 
	# updated in this function
	isBodyDebris = True
	
	# If classifying debris as all non-active particles
	if shouldDebrisBeAssignedAsAllNonActiveBodies:
		if bodyIndexAtInitialSnapshot < initialNumberOfActiveBodies:
			isBodyDebris = False

	# Otherwise, classifying debris as all bodies below a certain mass
	else:
		bodyMass_mSun = body.m
		bodyMass_mEarth = bodyMass_mSun * mSun_mEarth	

		if bodyMass_mEarth > massDivideBetweenDebrisAndNonDebris_mEarth:
			isBodyDebris = False
	
	return isBodyDebris
			
#------------------------------------------------------------------------
def GetSmallBodyParsAndOtherBodies(initialSnapshot):
	'''Function to identify debris bodies and get information about them.
	
	INPUTS:
		- initialSnapshot: The simulation at the initial time
	
	OUTPUTS:
		- smallA0sByHash_au: Dictionary of the small body initial 
			semimajor axes, in au (key = small body hash)
		- smallE0sByHash: Dictionary of the small body eccentricities
			(key = small body hash)
		- hashesInA0Order: A list of the particle hashes in 
			ascending order of initial semimajor axis
		- debrisBodies: list of Rebound objects in the initial 
			snapshot that are identified as debris
		- nonDebrisBodies: list of Rebound objects in the initial 
			snapshot that are not identified as debris
		- doesEachBodyHaveUniqueHash: boolean for whether each body has a
			unique hash
	'''

	print('Attempting to classify objects in simulation:')
	
	if shouldDebrisBeAssignedAsAllNonActiveBodies:
		print('	 Any non-active body at the initial snapshot will be classed as debris')
	else:
		print('	 Any object with mass at or below %s MEarth will be classed as debris' % GetNumberStringInScientificNotation(massDivideBetweenDebrisAndNonDebris_mEarth))
	print()
	
	# Initialize the dictionaries and lists
	smallA0sByHash_au, smallE0sByHash = {},{}
	debrisBodies, nonDebrisBodies = [], []
	
	# Loop through each body in the simulation, to identify debris and
	# non-debris bodies
	setOfUniqueHashValues = set()
	
	for bodyIndexAtInitialSnapshot in range(initialSnapshot.N):

		bodyAtInitialSnapshot = initialSnapshot.particles[bodyIndexAtInitialSnapshot]
		bodyHashValue = bodyAtInitialSnapshot.hash.value
		
		isBodyDebris = GetIfBodyIsDebris(bodyAtInitialSnapshot, bodyIndexAtInitialSnapshot, initialSnapshot.N_active)
		
		if isBodyDebris: debrisBodies.append(bodyAtInitialSnapshot)
		else: nonDebrisBodies.append(bodyAtInitialSnapshot)

		setOfUniqueHashValues.add(bodyHashValue)
		
	# Determine if each body has a unique hash
	if len(setOfUniqueHashValues) < len(debrisBodies) + len(nonDebrisBodies):
		doesEachBodyHaveUniqueHash = False
	else:
		doesEachBodyHaveUniqueHash = True

	# For each debris body, get its initial parameters
	for debrisBody in debrisBodies:
	
		# Get semimajor axis and eccentricity
		a0_au = initialSnapshot.particles[reboundHash(debrisBody.hash)].a
		e0 = initialSnapshot.particles[reboundHash(debrisBody.hash)].e

		# Set the dictionary values, with the key being the small body hash
		smallA0sByHash_au[debrisBody.hash.value] = a0_au
		smallE0sByHash[debrisBody.hash.value] = e0	
	
	# Get a list of the particle hashes in ascending order of initial 
	# semimajor axis
	hashesInA0Order = sorted(smallA0sByHash_au, key=smallA0sByHash_au.get)

	return smallA0sByHash_au, smallE0sByHash, hashesInA0Order, debrisBodies, nonDebrisBodies, doesEachBodyHaveUniqueHash

#--------------------------------------------------------------------------------------------------------------------------------
def CreateStirredParticlesPlot(particleBins,a0BinUpperEdges_au,simFilePath):
	'''Create the graph for the total number of particles stirred over 
	all of the simulations, given the amount of stirred/unstirred 
	particles per bin, and the divisions between the bins.
	
	INPUTS:
		- particleBins: array containing info on particles at each bin, 
			with numbers of how many particles in each bin are stirred, 
			unstirred, or scattered
		- a0BinUpperEdges_au: array of the semimajor axis 
			values for each bin upper edge
		- simFilePath: path string to the sim file to analyse
	'''

	print('Making plot...')
	
	# Plot a bar graph for the number of particles above unstirred 
	# eccentricity
	fig,ax = plt.subplots(1,1, figsize=(8,6))
	barWidth = a0BinUpperEdges_au[1]-a0BinUpperEdges_au[0]

	# Add title for sim file path
	titleString = '%s' % simFilePath
	titleString += '\nSnapshot interval: %s' % snapshotInterval
	fig.suptitle(titleString)
	
	# Unpack the particle bin data
	numbersStirred, numbersScattered, numbersUnstirred, percentsStirredOrScattered = [],[],[],[]
	
	for a0BinIndex in range(len(a0BinUpperEdges_au)):
		
		numberStirred = particleBins[a0BinIndex]['Stirred']
		numberScattered = particleBins[a0BinIndex]['Scattered']
		numberUnstirred = particleBins[a0BinIndex]['Unstirred']
		
		total = numberStirred + numberScattered + numberUnstirred
		
		percentStirredOrScattered = np.nan
		
		if total > 0:
			percentStirredOrScattered = 100*float(numberStirred+numberScattered)/total
		
		numbersStirred.append(numberStirred)
		numbersScattered.append(numberScattered)
		numbersUnstirred.append(numberUnstirred)
		percentsStirredOrScattered.append(percentStirredOrScattered)
		
	# Plot the rectangles for stirred, scattered, and unstirred particles.
	# Note the bar width is supplied as negative, because the bin upper 
	# edges (rather than lower) have been passed to the function
	rects1 = ax.bar(a0BinUpperEdges_au, numbersStirred, -barWidth, color='b', edgecolor='k', label="Stirred", align='edge')
	rects2 = ax.bar(a0BinUpperEdges_au, numbersScattered, -barWidth, color='g', edgecolor='k', label="Scattered", bottom=numbersStirred, align='edge')
	rects3 = ax.bar(a0BinUpperEdges_au, numbersUnstirred, -barWidth, color='#DC8019', edgecolor='k', label="Unstirred", bottom=list(np.array(numbersStirred)+np.array(numbersScattered)), align='edge')

	# Format the plot
	ax.set_xlabel("Initial semimajor axis / au")
	ax.set_ylabel("Initial number of debris bodies")
	ax.legend(loc='upper right', title='Labels: % stirred or scattered')

	# Get height and width to print each stir percentage, and the 
	# stirring percentages of the disk for each bin
	textHeight = []
	textWidth = []

	for semimajorAxisIndex in range(len(a0BinUpperEdges_au)):
		rectangleHeight = (rects1[semimajorAxisIndex].get_height() + rects2[semimajorAxisIndex].get_height() + rects3[semimajorAxisIndex].get_height())*1.01
		textHeight.append(rectangleHeight)
		textWidth.append(rects1[semimajorAxisIndex].get_x()+rects1[semimajorAxisIndex].get_width()/2.) 

	# Label the bars with the stir percentage above each bar
	LabelStirringPercentagesOnPlot(textHeight,textWidth,ax,percentsStirredOrScattered)

#-------------------------------------------------------------------------------------------------------------------------------
def LabelStirringPercentagesOnPlot(height,width,ax,percent):
	'''Label the stirring percentage values above each bin.
	
	INPUTS:
		- height: an array for the height to plot the text at
		- width: an array for the widths to plot the text at
		- ax: the axis of the plot to put the labels on
		- percent: an array of percentages, each value for one bin
	'''

	for i in range(len(percent)):
		ax.text(width[i], height[i], "%.1f %%" % percent[i], ha='center', va='bottom')

#--------------------------------------------------------------------------------------------------------------------------------
def CheckStirringConditions(particle1, particle2, a0s_au):
	''' Check that the particles meet the desired stirring criteria, 
	based on their orbital parameters.
	
	INPUTS:
		- particle1: first particle to consider
		- particle2: second particle to consider
		- a0s_au: initial semimajor axis 
			dictionary (key = particle hash), with values in au
	
	OUTPUTS:
		- areParticlesStirred: True if all conditions passed (and therefore
			compare the particles to see if they could undergo a 
			destructive collision), False otherwise
	'''

	# Start with stirring condition False, only change if all conditions 
	# passed
	areParticlesStirred = False

	# Get the orbits for each particle
	orbit1 = particle1.calculate_orbit()
	orbit2 = particle2.calculate_orbit()

	# Check all condiitions - if they all pass, change the check_stirring
	#  variable to True and check if these particles could undergo a 
	# destructive collision. Note that the initial particle semimajor 
	# axes are *not* checked against each other here, because that test
	# is done elsewhere
	if orbit1.a > 0 and orbit2.a > 0:   # orbits are bound
		if np.abs(orbit2.a - orbit1.a)/(0.5*(orbit1.a+orbit2.a)) <= particleNearRange_percent/100:  # particles within certain values of each other now
			#if np.abs(a0s_au[particle1.hash.value] - a0s_au[particle2.hash.value])/(0.5*(a0s_au[particle1.hash.value]+a0s_au[particle2.hash.value])) < particleNearRange_percent/100: # particles initially in range of each other
			if np.abs(orbit1.inc - orbit2.inc) < inclinationRange_deg * np.pi / 180: # particles within the inclination range of each other now
				if np.abs(orbit1.a - a0s_au[particle1.hash.value])/a0s_au[particle1.hash.value] < particleInitialRange_percent/100: # particle1 hasnt moved far from initial spot (redundant to also check particle2)
					areParticlesStirred = True

	return areParticlesStirred

#---------------------------------------------------------------------------------------------------------------------------------
def CalculateCollisionVelocities_mPerS(particle1,particle2,centralMass_mSun):
	''' Determine if the orbits of two particles cross, and if so
	calculate the collision speed between the particles.
	
	INPUTS:
		- particle1: first particle to consider
		- particle2: second particle to consider
		- centralMass_mSun: total mass of the central body (or bodies)
	
	OUTPUTS:
		- collisionSpeed1_mPerS: first solution to the colliding 
			speed of the two particles (0 if orbits don't cross)
		- collisionSpeed2_mPerS: second solution to the colliding 
			speed of the two particles (0 if orbits don't cross)
	'''

	# Calculate the orbits of each particle
	orbit1 = particle1.calculate_orbit()
	orbit2 = particle2.calculate_orbit()

	# Grab some particle parameters, starting with eccentricities
	e1 = orbit1.e
	e2 = orbit2.e

	# Semimajor axes
	a1_au = orbit1.a
	a2_au = orbit2.a

	# Longitude of pericentre
	pOmega1_rad = particle1.pomega
	pOmega2_rad = particle2.pomega
	deltaPOmega_rad = pOmega1_rad-pOmega2_rad

	# Calculate the semilatus rectum of each orbit
	p1_au = a1_au * (1-e1**2)
	p2_au = a2_au * (1-e2**2)

	# Calculate constants and the orbit crossing condition
	A_au = p2_au-p1_au
	B_au = e1*p2_au*np.cos(deltaPOmega_rad) - e2*p1_au
	C_au = e1*p2_au*np.sin(deltaPOmega_rad)
	orbitCrossingCondition_au2 = C_au**2 + B_au**2 - A_au**2

	# If the orbit crossing condition is met, the orbits will cross and
	# can check the velocities
	if orbitCrossingCondition_au2 >= 0:

		cosTheta1 = (-A_au*B_au + C_au*np.sqrt(orbitCrossingCondition_au2))/(B_au**2+C_au**2)
		cosTheta2 = (-A_au*B_au - C_au*np.sqrt(orbitCrossingCondition_au2))/(B_au**2+C_au**2)

		theta1_rad = GetThetaFromCosThetaInCorrectQuadrant(cosTheta1, A_au, B_au, C_au)
		theta2_rad = GetThetaFromCosThetaInCorrectQuadrant(cosTheta2, A_au, B_au, C_au)

		# Calculate the value of r (the intersection point) for each 
		# solution
		r1_au = p2_au/(1+e2*np.cos(theta1_rad))
		r2_au = p2_au/(1+e2*np.cos(theta2_rad))

		# Calculate the two solutions for the collision speed of the 
		# two particles
		collisionSpeed1_mPerS = np.sqrt(6.67e-11*centralMass_mSun*2e30/(1.5e11)) * np.sqrt((e1*np.sin(theta1_rad-deltaPOmega_rad)/np.sqrt(p1_au) - e2*np.sin(theta1_rad)/np.sqrt(p2_au))**2 + (np.sqrt(p1_au)/r1_au - np.sqrt(p2_au)/r1_au)**2)
		collisionSpeed2_mPerS = np.sqrt(6.67e-11*centralMass_mSun*2e30/(1.5e11)) * np.sqrt((e1*np.sin(theta2_rad-deltaPOmega_rad)/np.sqrt(p1_au) - e2*np.sin(theta2_rad)/np.sqrt(p2_au))**2 + (np.sqrt(p1_au)/r2_au - np.sqrt(p2_au)/r2_au)**2)
	
	# If orbits didn't cross, no collision (so set velocities to 0)
	else:
		collisionSpeed1_mPerS = 0
		collisionSpeed2_mPerS = 0

	return collisionSpeed1_mPerS, collisionSpeed2_mPerS

#------------------------------------------------------------------------
def GetThetaFromCosThetaInCorrectQuadrant(cosTheta, A_au, B_au, C_au):
	'''Get theta from cosTheta. Takes account of the correct quadrant'''
	
	# Calculate sin(theta)
	sinTheta = (-A_au-B_au*cosTheta) / C_au

	# Get solution to cosTheta in 0 < theta < pi
	arccosTheta0ToPi_rad = np.arccos(cosTheta)
	
	# Determine which quadrant theta is in
	if sinTheta >= 0:
		theta_rad = arccosTheta0ToPi_rad
	
	else:
		theta_rad = -arccosTheta0ToPi_rad	

	return theta_rad

#------------------------------------------------------------------------
def PrintSimParameters(debrisBodies, nonDebrisBodies):
	'''Print out the parameters inferred from the simulation, so the user
	can check the correct particles are inferred as debris.
	
	INPUTS:
		- debrisBodies: list of Rebound objects in the initial 
		 snapshot that are identified as debris
		- nonDebrisBodies: list of Rebound objects in the initial 
		 snapshot that are not identified as debris
	'''

	# Get the total mass in the non-debris bodies
	nonDebrisTotalMass_mSun = 0

	for nonDebrisBody in nonDebrisBodies:
			nonDebrisTotalMass_mSun += nonDebrisBody.m

	# Get information about the debris bodies
	debrisTotalMass_mSun = 0

	for debrisBody in debrisBodies:
			debrisTotalMass_mSun += debrisBody.m	

	debrisTotalMass_mEarth = debrisTotalMass_mSun*mSun_mEarth
	
	print('Bodies inferred from initial simulation snapshot:')
	print('	 - %s non-debris bodies, with total mass %s MSun' % (len(nonDebrisBodies), GetNumberStringInScientificNotation(nonDebrisTotalMass_mSun)))
	print('	 - %s debris bodies, with total mass %s MSun (%s MEarth)' % (len(debrisBodies), GetNumberStringInScientificNotation(debrisTotalMass_mSun), GetNumberStringInScientificNotation(debrisTotalMass_mEarth)))

	print()

#------------------------------------------------------------------------
def GetIfUserInputsHaveCorrectFormat():
	'''Determine if the user input values have the correct formats and 
	ranges.
	
	OUTPUTS:
		- doUserInputsHaveCorrectFormat: boolean for whether inputs 
			correctly supplied
		- errorReason: string, describing why inputs incorrectly
			supplied. If correctly supplied, then None
	'''
			
	doUserInputsHaveCorrectFormat = False

	# Define user inputs by name
	userInputs = {  'stirredParticleSize_m': stirredParticleSize_m,
					'numberSemimajorAxisBins': numberSemimajorAxisBins,
					'snapshotInterval': snapshotInterval,
					'particleNearRange_percent': particleNearRange_percent,
					'particleInitialRange_percent': particleInitialRange_percent,
					'inclinationRange_deg': inclinationRange_deg,
					'shouldPlotBeMade': shouldPlotBeMade,
					'shouldPlotBeSaved': shouldPlotBeSaved,
					'shouldDataBeSaved': shouldDataBeSaved}

	# Add optional arguments if necessary
	if shouldDebrisBeAssignedAsAllNonActiveBodies:
		userInputs['shouldDebrisBeAssignedAsAllNonActiveBodies'] = shouldDebrisBeAssignedAsAllNonActiveBodies
	else:
		 userInputs['massDivideBetweenDebrisAndNonDebris_mEarth'] = massDivideBetweenDebrisAndNonDebris_mEarth

	if shouldPlotBeSaved:
		userInputs['outputPlotFilePath'] = outputPlotFilePath
		
	if shouldDataBeSaved:
		userInputs['outputCSVOutcomesByA0BinFilePath'] = outputCSVOutcomesByA0BinFilePath
		userInputs['outputCSVTotalOutcomesFilePath'] = outputCSVTotalOutcomesFilePath
		
	# Go through each user input, and check it is in the right format
	for userInputName in userInputs:
		userInputValue = userInputs[userInputName]
		userInputType = type(userInputValue)
		
		# Check inputs that must be either ints or floats
		if userInputName in ['stirredParticleSize_m', 'particleNearRange_percent', 'particleInitialRange_percent', 'inclinationRange_deg', 'massDivideBetweenDebrisAndNonDebris_mEarth'] and isinstance(userInputValue, int) == False and isinstance(userInputValue, float) == False:
			errorReason = '%s must be int or float, not %s' % (userInputName, userInputType)
			return doUserInputsHaveCorrectFormat, errorReason
						
		# Check inputs that must be ints
		if userInputName in ['numberSemimajorAxisBins', 'snapshotInterval'] and isinstance(userInputValue, int) == False:
			errorReason = '%s must be int, not %s' % (userInputName, userInputType)
			return doUserInputsHaveCorrectFormat, errorReason 
		
		# Check inputs that must be boolean
		if userInputName in ['shouldPlotBeMade', 'shouldPlotBeSaved', 'shouldDataBeSaved', 'shouldDebrisBeAssignedAsAllNonActiveBodies'] and isinstance(userInputValue, bool) == False:
			errorReason = '%s must be boolean, not %s' % (userInputName, userInputType)
			return doUserInputsHaveCorrectFormat, errorReason
		
		# Check inputs that must be strings
		if userInputName in ['outputPlotFilePath', 'outputCSVOutcomesByA0BinFilePath', 'outputCSVTotalOutcomesFilePath'] and isinstance(userInputValue, str) == False:
			errorReason = '%s must be string, not %s' % (userInputName, userInputType)
			return doUserInputsHaveCorrectFormat, errorReason
				
		# Check values that must be greater than zero
		if userInputName in ['stirredParticleSize_m', 'numberSemimajorAxisBins', 'snapshotInterval', 'particleNearRange_percent', 'particleInitialRange_percent', 'inclinationRange_deg', 'massDivideBetweenDebrisAndNonDebris_mEarth'] and userInputValue <= 0:
			errorReason = '%s must be greater than zero' % (userInputName)
			return doUserInputsHaveCorrectFormat, errorReason

		# Check degrees
		if userInputName in ['inclinationRange_deg'] and userInputValue > 180:
			errorReason = '%s must be <= 180 degrees' % (userInputName)
			return doUserInputsHaveCorrectFormat, errorReason

	# If this point reached, then all user inputs have the correct format
	doUserInputsHaveCorrectFormat = True
	errorReason = None

	return doUserInputsHaveCorrectFormat, errorReason
	
#------------------------------------------------------------------------
def SaveAnalysisDataAsCSV(a0BinEdges_au, totalSmallsByA0BinIndexAndOutcome, totalSmallsByOutcome):
	'''Save the stirring-analysis data as CSV files.
	
	INPUTS:
		- a0BinEdges_au: array of the semimajor axis values 
			for each bin edge (both upper and lower edges)
		- totalSmallsByA0BinIndexAndOutcome: Nested dictionary
			containing the number of stirred, scattered and unstirred 
			debris bodies by the index of the initial semimajor-axis bin
		- totalSmallsByOutcome: Dictionary containing the number of
			stirred, scattered and unstirred debris bodies
	'''

	# Exit if the file path is invalid
	for outputCSVFilePath in [outputCSVOutcomesByA0BinFilePath, outputCSVTotalOutcomesFilePath]:
		canCSVBeSaved = GetIfCanSaveToFilePath(outputCSVFilePath, 'data')
		
		if canCSVBeSaved == False:
			return

	# Save the first file, which is the total number of debris bodies
	# undergoing each outcome, separated by their initial semimajor axes
	with open(outputCSVOutcomesByA0BinFilePath, 'w') as stirringOutcomeByInitialAFile:

		# Write the column headers
		stirringOutcomeByInitialAFile.write('semimajorAxisBinLowerEdge_au, semimajorAxisBinUpperEdge_au, semimajorAxisBinCentre_au, numberStirred, numberUnstirred, numberScattered, percentStirred, percentUnstirred, percentScattered')

		# Write the data to the file
		for semimajorAxisBinEdgeIndex in range(len(a0BinEdges_au)-1):
			binLowerEdge_au = a0BinEdges_au[semimajorAxisBinEdgeIndex]
			binUpperEdge_au = a0BinEdges_au[semimajorAxisBinEdgeIndex+1]
			binCentre_au = 0.5*(binLowerEdge_au+binUpperEdge_au)

			numberStirred = totalSmallsByA0BinIndexAndOutcome[semimajorAxisBinEdgeIndex]['Stirred']
			numberUnstirred = totalSmallsByA0BinIndexAndOutcome[semimajorAxisBinEdgeIndex]['Unstirred']
			numberScattered = totalSmallsByA0BinIndexAndOutcome[semimajorAxisBinEdgeIndex]['Scattered']

			numberInitiallyInBin = numberStirred+numberUnstirred+numberScattered

			percentStirred, percentUnstirred, percentScattered = np.nan, np.nan, np.nan
							
			if numberInitiallyInBin > 0:
				percentStirred = round(100*float(numberStirred) / numberInitiallyInBin, 1)
				percentUnstirred = round(100*float(numberUnstirred) / numberInitiallyInBin, 1)
				percentScattered = round(100*float(numberScattered) / numberInitiallyInBin, 1)		

			stirringOutcomeByInitialAFile.write('\n%s, %s, %s, %s, %s, %s, %s, %s, %s' % (round(binLowerEdge_au,1), round(binUpperEdge_au,1), round(binCentre_au,1), numberStirred, numberUnstirred, numberScattered, percentStirred, percentUnstirred, percentScattered))

	print('Debris outcome data by semimajor axis saved as %s' % outputCSVOutcomesByA0BinFilePath)
	
	# Save the second file, which is the total number of debris bodies
	# undergoing each outcome, regardless of semimajor axis
	with open(outputCSVTotalOutcomesFilePath, 'w') as stirringOutcomeFile:

		# Write the column headers
		stirringOutcomeFile.write('numberStirred, numberUnstirred, numberScattered, percentStirred, percentUnstirred, percentScattered')

		# Write the data to the file
		numberStirred = totalSmallsByOutcome['Stirred']
		numberUnstirred = totalSmallsByOutcome['Unstirred']
		numberScattered = totalSmallsByOutcome['Scattered']

		numberInitially = numberStirred+numberUnstirred+numberScattered
		
		percentStirred = round(100*float(numberStirred) / numberInitially, 1)
		percentUnstirred = round(100*float(numberUnstirred) / numberInitially, 1)
		percentScattered = round(100*float(numberScattered) / numberInitially, 1)	  
		
		stirringOutcomeFile.write('\n%s, %s, %s, %s, %s, %s' % (numberStirred, numberUnstirred, numberScattered, percentStirred, percentUnstirred, percentScattered))

	print('Debris total outcome data saved as %s' % outputCSVTotalOutcomesFilePath)
	print()
	
#------------------------------------------------------------------------
def GetIfCanSaveToFilePath(filePathString, plotOrDataString):
	'''Get whether the file path provided for an output file is valid.
	Used to check validity before saving outputs, to prevent crashes.
	
	INPUTS:
		- filePathString: the desired output path
		- plotOrDataString: either 'plot' or 'data', to describe output

	OUTPUTS:
		- canSaveToFilePath: Boolean for whether path is valid
	'''
	
	canSaveToFilePath = True
	
	# Get index of final dot, for filename
	indexOfLastDot = filePathString.rfind('.')
	
	# Get index of final slash, for directory
	indexOfLastSlash = filePathString.rfind('/')

	# Check that this is a valid file location; there must be a dot for
	# an extension, and it must occur after the final slash (directory)
	if indexOfLastDot == -1 or indexOfLastDot <= indexOfLastSlash:
		canSaveToFilePath = False
		reasonCannotSave = '%s is not a valid path for the output file (missing extention or invalid format)' % filePathString

	# Check that the save directory exists
	elif indexOfLastSlash >= 0:
		directoryPath = filePathString[:indexOfLastSlash]
		
		if os.path.isdir(directoryPath) == False:
			canSaveToFilePath = False
			reasonCannotSave = 'directory %s does not exist' % directoryPath
	
	if canSaveToFilePath == False:
		print('	 ***WARNING: %s. No %s will be saved ***' % (reasonCannotSave, plotOrDataString))
		
	return canSaveToFilePath

#------------------------------------------------------------------------
def GetSimFilePath():
	'''Read in the simulation archive file (supplied from the terminal 
	command line) to analyse'''

	simFilePath = None
	wasInputCorrectlySupplied = False
		
	# Proceed only if correct number of arguments supplied
	if len(sys.argv[1:]) == 1:

		argString = sys.argv[1]

		# Check the argument is a path, and that it exists
		if os.path.exists(argString):

			# Check the argument is an archive file
			if argString[-4:] == '.bin':
				wasInputCorrectlySupplied = True
				simFilePath = argString

			# Otherwise the argument is not a .bin file
			else:
				errorReason = '%s is not a .bin file' % argString

		# Otherwise the path does not exist
		else:
			errorReason = 'file %s does not exist' % argString

	# Otherwise incorrect number of arguments supplied
	else:
		errorReason = 'Code takes exactly 1 argument: the path of one .bin file to be analysed'

	# Print error if arguments incorrectly supplied
	if wasInputCorrectlySupplied == False:
		print('***Error: %s ***' % errorReason)

	return simFilePath, wasInputCorrectlySupplied

#------------------------------------------------------------------------
def GetNumberStringInScientificNotation(num, precision=None, exponent=None):
	'''Code by Tim Pearce. Returns a string representation of the 
	scientific notation of the given number, with specified precision
	(number of decimal digits to show). The exponent to be used can also
	be specified explicitly.
	
	Adapted from https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting.'''

	# Catch nans
	if math.isnan(num):	return np.nan
		
	# Default number of decimal digits (in case precision unspecified)
	defaultDecimalDigits = 2

	# Get precision if not defined
	if precision is None:
		precision = defaultDecimalDigits
			
	# Get exoponent if not defined
	if exponent is None:
	
		# Catch case if number is zero
		if num == 0:
			exponent = 0
		
		# Otherwise number is non-zero
		else:
			exponent = int(math.floor(np.log10(abs(num))))
	
	# Get the coefficient
	coefficient = round(num / float(10**exponent), precision)
	
	# Adjust if rounding has taken coefficient to 10
	if coefficient == 10:
		coefficient = 1
		exponent += 1
	
	# Get the output string
	outString = '{0:.{2}f}e{1:d}'.format(coefficient, exponent, precision)

	return outString

#------------------------------------------------------------------------
def GetEffectiveCentralMass_mSun(particle):
	'''Get the effective central mass of the system. Note this is not
	necessarily the mass of a central star, if there is e.g. a binary
	system or massive planets.
	
	INPUTS:
		- particle: a Rebound particle in the simulation
	
	OUTPUTS:
		- effectiveCentralMass_mSun: effective central mass, in solar 
			masses
	'''
	
	orbit = particle.calculate_orbit()
	
	a_au = orbit.a
	period_yr = orbit.P
	
	effectiveCentralMass_mSun = a_au**3 / period_yr**2
	
	return effectiveCentralMass_mSun
	
#------------ END OF HELPER FUNCTIONS, START OF MAIN FUNCTION -----------

def RunStirringCheck(simFilePath):
	'''Main function to check the stirring done in a planet+projectile 
	simulation, a planet-only simulation, and comparing the stirring done
	between the two.

	INPUTS:
		- simFilePath: path string to the sim file to analyse
	'''
	# Get the simulation archive from the simulation file
	print('Loading simulation archive...')
	simArchive = rebound.SimulationArchive(simFilePath)

	initialSnapshot = simArchive[0]
	finalSnapshot = simArchive[-1]

	print('Loading complete')
	print()
			
	# Calculate the fragmentation speed based on the grain size
	fragmentationSpeed_mPerS = GetFragmentationSpeedForBasalt(stirredParticleSize_m)
	
	# Define the range of simulations to check, as well as the initial 
	# and final snapshot numbers. Note that the final snapshot is always
	# checked
	finalSnapshotIndex = len(simArchive) -1
	snapshotIndicesToCheckInDescendingOrder = []
	
	snapshotIndexToCheck = finalSnapshotIndex
	while snapshotIndexToCheck >= 0:
		snapshotIndicesToCheckInDescendingOrder.append(snapshotIndexToCheck)
		snapshotIndexToCheck -= snapshotInterval

	# If sim units are not (au, yr, mSun), exit
	units = initialSnapshot.units
	if units['length'] != 'au' or units['mass'] != 'msun' or units['time'] != 'yr':
		print('***ERROR: Rebound simulation does not have units of au, yr, mSun ***')
		print()
		return
		
	# Get the initial particle semimajor axes and eccentricities
	smallA0sByHash_au, smallE0sByHash, unstirredHashesInA0Order, debrisBodies, nonDebrisBodies, doesEachBodyHaveUniqueHash = GetSmallBodyParsAndOtherBodies(initialSnapshot)

	# Print the inferred simulation parameters
	PrintSimParameters(debrisBodies, nonDebrisBodies)
	
	# If particles don't have unique hashes, then exit
	if doesEachBodyHaveUniqueHash == False:
		print("***ERROR: Bodies in Rebound simulation do not have unique hashes (This version requires unique hashing, i.e. for each particle to have been initilised with e.g. sim.add(hash=X, ...), where X is a unique string or value. Future versions of this code may work without hashing) ***")
		print()
		return	
	
	# If not enough debris detected, exit
	if len(smallA0sByHash_au) < 2:
		
		if len(smallA0sByHash_au) <= 0:
			reasonString = 'no debris detected'
		
		elif len(smallA0sByHash_au) == 1:
			reasonString = 'require at least two debris bodies'
					 
		print('***ERROR: %s in initial simulation snapshot (do you need to adjust shouldDebrisBeAssignedAsAllNonActiveBodies or massDivideBetweenDebrisAndNonDebris_mEarth?) ***' % reasonString)
		print()
				
		return

	# Define bins for initial semimajor axes. Define just outside max and 
	# min values to avoid errors when bodies lie exactly at inner/outer 
	# bin edges
	a0BinEdges_au = np.linspace(0.999999*min(smallA0sByHash_au.values()), 1.000001*max(smallA0sByHash_au.values()), numberSemimajorAxisBins+1)
	a0BinUpperEdges_au = a0BinEdges_au[1:]
	
	# Get the initial number of particles per bin before anything has 
	# been scattered/removed from system
	initialParticlesPerBin = {}
	
	for a0BinIndex in range(len(a0BinUpperEdges_au)):
		initialParticlesPerBin[a0BinIndex] = 0

	for debrisBody in debrisBodies:
		a0_au = smallA0sByHash_au[debrisBody.hash.value]
		a0BinIndex = np.digitize(a0_au, a0BinEdges_au) -1
		initialParticlesPerBin[a0BinIndex] += 1 

	# Create a dictionary to record the number of stirred, unstirred
	# and scattered particles
	totalSmallsByA0BinIndexAndOutcome = {}
	for a0BinIndex in range(len(a0BinUpperEdges_au)):
		 totalSmallsByA0BinIndexAndOutcome[a0BinIndex] = {'Stirred':0, 'Scattered':0, 'Unstirred':0}

	# For each small body, calculate some values based on thier initial 
	# semimajor axis
	parsFromA0ByParticleHash = {}
	
	for smallHash in smallA0sByHash_au:
		parsFromA0ByParticleHash[smallHash] = {}
		
		a0_au = smallA0sByHash_au[smallHash]
		
		# Largest a0 of another particle to consider when doing stirring 
		# tests
		parsFromA0ByParticleHash[smallHash]['maxA0ToConsiderNearParticle_au'] = a0_au*(1.+particleInitialRange_percent/100.)

		# Index of initial semimajor axis bin
		parsFromA0ByParticleHash[smallHash]['a0BinIndex'] = np.digitize(a0_au, a0BinEdges_au) -1
		
	# Print when the system started checking the stirring
	startTimeString = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print('Running stirring analysis (started at %s)...' % startTimeString)
	
	codeStartTime_s = systime.time()
	
	# Loop through each snapshot at the chosen interval, checking for 
	# stirring between particles at each snapshot. Analyse snapshots in
	# reverse order, because stirring level would increase over time so
	# final snapshot likely to have highest stirring level; this 
	# increases the code efficiency
	snapshotsAnalysed = 0

	for snapshotIndex in snapshotIndicesToCheckInDescendingOrder:
		
		# Retrieve the simulation snapshot to check
		snapshot = simArchive[snapshotIndex]

		# If any debris particles that were in the initial snapshot no
		# longer exist at this snapshot, then assume they were scattered
		# and removed. Remove them from bodies to check for stirring.
		# Uses 'while True' because list is updated during loop
		indexOfFirstUncheckedParticle = 0
		
		while True:

			# If all particles have been checked, break the loop
			if indexOfFirstUncheckedParticle == len(unstirredHashesInA0Order):
				break

			particleHash = unstirredHashesInA0Order[indexOfFirstUncheckedParticle]
			particleA0_au = smallA0sByHash_au[particleHash] 
			shouldParticleBeRemoved = False
			
			try:
				particle = snapshot.particles[reboundHash(particleHash)]

			# If particle is not in the simulation snapshot, then it has 
			# been scattered. Target for removal
			except rebound.ParticleNotFound:
				shouldParticleBeRemoved = True
			
			# Alternatively, particle may still exist but still have been
			# scattered (just not removed). Flag these particles too
			if shouldParticleBeRemoved == False:
				orbit = particle.calculate_orbit()
				if orbit.e >= 1 or 100.*abs(float(orbit.a)/particleA0_au - 1.) >= minSemimajorAxisChangeToBeScattered_percent:
					shouldParticleBeRemoved = True
			
			# Remove particle if scattered. After removal, the next 
			# particle will have had its list index reduced by 1, so the 
			# index to check will remain the same
			if shouldParticleBeRemoved:
				a0BinIndex = np.digitize(particleA0_au, a0BinEdges_au) -1		   
				totalSmallsByA0BinIndexAndOutcome[a0BinIndex]['Scattered'] += 1
				unstirredHashesInA0Order.remove(particleHash)

			# Otherwise the particle is not removed, so move on to the
			# next index
			else:
				indexOfFirstUncheckedParticle += 1

		# If there are no debris particles left, break
		if len(unstirredHashesInA0Order) == 0:
			break

		# Get the effective central mass
		exampleParticle = snapshot.particles[reboundHash(unstirredHashesInA0Order[0])]
		centralMass_mSun = GetEffectiveCentralMass_mSun(exampleParticle)

		# Loop through each unstirred particle. Done with 'while True' 
		# because elements will be continually removed from the list that 
		# we are looping over
		indexOfFirstUncheckedUnstirredParticle = 0

		while True:
			particle1Hash = unstirredHashesInA0Order[indexOfFirstUncheckedUnstirredParticle]
			particle1A0_au = smallA0sByHash_au[particle1Hash]

			isParticle1Stirred = False

			# Only particles with initial semimajor axes close to the 
			# initial value for Particle1 will be considered
			largestA0ToConsiderNearParticle1_au = parsFromA0ByParticleHash[particle1Hash]['maxA0ToConsiderNearParticle_au']

			# Check against each particle with an initial semimajor axis 
			# above this one, which starts close to this one
			for indexOfUnstirredParticle2 in range(indexOfFirstUncheckedUnstirredParticle+1, len(unstirredHashesInA0Order)):
				particle2Hash = unstirredHashesInA0Order[indexOfUnstirredParticle2]
				particle2A0_au = smallA0sByHash_au[particle2Hash]

				# If these particles are initially too far away in 
				# initial semimajor-axis space, then move on to the next
				# particle1
				if particle2A0_au > largestA0ToConsiderNearParticle1_au:
					break
				
				# Otherwise, these particles are sufficiently close in 
				# inital semimajor-axis. Run the stirring check
				else:

					# See if the particles meet the stirring conditions
					particle1 = snapshot.particles[reboundHash(particle1Hash)]
					particle2 = snapshot.particles[reboundHash(particle2Hash)]
					
					if CheckStirringConditions(particle1,particle2,smallA0sByHash_au):
					
						# Calculate the collision velocities of the two
						# particles (if orbits don't cross, velocities = 0)
						collisionSpeed1_mPerS, collisionSpeed2_mPerS = CalculateCollisionVelocities_mPerS(particle1,particle2,centralMass_mSun)
							   
						# If the collisional speed is greater than the
						# fragmentation speed (and the orbits cross), 
						# consider the particles stirred. Remove both
						# from the list of particles to consider
						if collisionSpeed1_mPerS > fragmentationSpeed_mPerS or collisionSpeed2_mPerS > fragmentationSpeed_mPerS:

							# Update the total number of stirred 
							# particles in the initial semimajor-axis 
							# bins, and remove particle2 (particle1 will 
							# be removed later)
							a0BinIndex2 = parsFromA0ByParticleHash[particle2Hash]['a0BinIndex']

							totalSmallsByA0BinIndexAndOutcome[a0BinIndex2]['Stirred'] += 1
							unstirredHashesInA0Order.remove(particle2Hash)							
						   
							isParticle1Stirred = True

							break

			# Remove particle1 if it is stirred. In this case, the next 
			# particle1 will have the same index in the list as this one,
			# so the index value is not updated
			if isParticle1Stirred:
			
				a0BinIndex1 = parsFromA0ByParticleHash[particle1Hash]['a0BinIndex']							

				totalSmallsByA0BinIndexAndOutcome[a0BinIndex1]['Stirred'] += 1
				unstirredHashesInA0Order.remove(particle1Hash)							

			# Otherwise, particle 1 is not stirred. The next particle1 
			# will be considered
			else:
				indexOfFirstUncheckedUnstirredParticle += 1
			
			# If all particles have now been checked, break
			if indexOfFirstUncheckedUnstirredParticle == len(unstirredHashesInA0Order):
				break

		# Print snapshot progress
		snapshotsAnalysed += 1
		print('	 Snapshot %s of %s analysed' % (snapshotsAnalysed, len(snapshotIndicesToCheckInDescendingOrder)), end='\r')

	# Print out snapshot and time information. Padding spaces are used so
	# the new line properly overlaps the previous 'Snapshot %s of %s
	# analysed' line
	paddingSpaces = '   ' + len(str(len(snapshotIndicesToCheckInDescendingOrder)))*' '
	print('	 %s snapshots analysed%s' % (len(snapshotIndicesToCheckInDescendingOrder), paddingSpaces))
	
	endTimeString = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	runTime_s = systime.time() - codeStartTime_s
	
	print('Analysis complete (finished at %s, run time: %s sec)' % (endTimeString, round(runTime_s)))
	print()
	
	# The number of unstirred particles should be the initial number per
	# bin less the number of stirred and scattered particles in each bin
	for semimajorAxisIndex in range(numberSemimajorAxisBins):
		totalSmallsByA0BinIndexAndOutcome[semimajorAxisIndex]['Unstirred'] = initialParticlesPerBin[semimajorAxisIndex] - totalSmallsByA0BinIndexAndOutcome[semimajorAxisIndex]['Stirred'] - totalSmallsByA0BinIndexAndOutcome[semimajorAxisIndex]['Scattered']

	# Get the total number of particles across all bins by outcome
	totalSmallsByOutcome = {'Unstirred': 0, 'Stirred': 0, 'Scattered': 0}
	
	for semimajorAxisIndex in totalSmallsByA0BinIndexAndOutcome:
		for outcome in totalSmallsByA0BinIndexAndOutcome[semimajorAxisIndex]:
			totalSmallsByOutcome[outcome] += totalSmallsByA0BinIndexAndOutcome[semimajorAxisIndex][outcome]
	
	print('Analysis results:')
	for outcome in totalSmallsByOutcome:
		print('	 %s debris bodies: %s (%s %%)' % (outcome.lower(), totalSmallsByOutcome[outcome], round(100*totalSmallsByOutcome[outcome]/float(len(debrisBodies)),2)))
	print()
	
	# If the toggle is set, save the data
	if shouldDataBeSaved:
		SaveAnalysisDataAsCSV(a0BinEdges_au, totalSmallsByA0BinIndexAndOutcome, totalSmallsByOutcome)

	# Create the stirring plot
	CreateStirredParticlesPlot(totalSmallsByA0BinIndexAndOutcome, a0BinUpperEdges_au, simFilePath)

	# If the toggle is set, save the figure
	if shouldPlotBeSaved:
		canPlotBeSaved = GetIfCanSaveToFilePath(outputPlotFilePath, 'plot')
		
		if canPlotBeSaved:
			plt.savefig(outputPlotFilePath)
			print('Plot saved as %s' % outputPlotFilePath)

	# If the toggle is set, display the figure
	if shouldPlotBeMade:
		plt.show()
	
############################ START OF PROGRAM ###########################
print()

# Get the sim file path from the command line
simFilePath, wasInputCorrectlySupplied = GetSimFilePath()

if wasInputCorrectlySupplied:
	print("Analyzing simulation %s" % simFilePath)
	print()

	# If user inputs are incorrectly set up, exit
	doUserInputsHaveCorrectFormat, errorReason = GetIfUserInputsHaveCorrectFormat()
	if doUserInputsHaveCorrectFormat == False:
		print('***ERROR: %s *** ' % errorReason)
		
	# Otherwise, run the analysis
	else:
		RunStirringCheck(simFilePath)

	print('Complete')
	
print()

#########################################################################

