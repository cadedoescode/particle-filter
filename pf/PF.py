import numpy as np
from Visualization import Visualization
from scipy.stats import norm
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PF(object):
    """A class for implementing particle filters

        Attributes
        ----------
        numParticles : The number of particles to use
        particles :    A 3 x numParticles array, where each column represents a
                       particular particle, i.e., particles[:,i] = [x^(i), y^(i), theta^(i)]
        weights :      An array of length numParticles array, where each entry
                       denotes the weight of that particular particle
        Alpha :        Vector of 6 noise coefficients for the motion model
                       (See Table 5.3 in Probabilistic Robotics)
        laser :        Instance of the laser class that defines LIDAR params,
                       observation likelihood, and utils
        gridmap :      An instance of the Gridmap class that specifies
                       an occupancy grid representation of the map
                       where 1: occupied and 0: free
        visualize:     Boolean variable indicating whether to visualize
                       the particle filter


        Methods
        -------
        sampleParticlesUniform : Samples a set of particles according to a
                                 uniform distribution
        sampleParticlesGaussian: Samples a set of particles according to a
                                 Gaussian distribution over (x,y) and a
                                 uniform distribution over theta
        getParticle :            Returns the (x, y, theta) and weight associated
                                 with a particular particle id.
        getNormalizedWeights :   Returns the normalized particle weights (numpy.array)
        getMean :                Queries the sample-based estimate of the mean
        prediction :             Performs the prediction step
        update :                 Performs the update step
        run :                    The main loop of the particle filter

    """

    def __init__(self, numParticles, Alpha, laser, gridmap, visualize=True):
        """Initialize the class

            Args
            ----------
            numParticles : The number of particles to use
            Alpha :        Vector of 6 noise coefficients for the motion model
                           (See Table 5.3 in Probabilistic Robotics)
            laser :        Instance of the laser class that defines LIDAR params,
                           observation likelihood, and utils
            gridmap :      An instance of the Gridmap class that specifies
                           an occupancy grid representation of the map
                           here 1: occupied and 0: free
            visualize:     Boolean variable indicating whether to visualize
                           the particle filter (optional, default: True)
        """
        self.numParticles = numParticles
        self.Alpha = Alpha
        self.laser = laser
        self.gridmap = gridmap
        self.visualize = visualize

        # particles is a numParticles x 3 array, where each column denote a particle_handle
        # weights is a numParticles x 1 array of particle weights
        self.particles = None
        self.weights = None
        self.resample_fn = None

        if self.visualize:
            self.vis = Visualization()
            self.vis.drawGridmap(self.gridmap)
        else:
            self.vis = None

    def sampleParticlesUniform(self):
        """
            Samples the set of particles according to a uniform distribution and
            sets the weights to 1/numParticles. Particles in collision are rejected
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi, np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0, (n-1)*self.gridmap.xres)
                y = np.random.uniform(0, (m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def sampleParticlesUniform2(self):
        (m, n) = self.gridmap.getShape()

        particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            theta = np.random.uniform(-np.pi, np.pi)
            inCollision = True
            while inCollision:
                x = np.random.uniform(0, (n-1)*self.gridmap.xres)
                y = np.random.uniform(0, (m-1)*self.gridmap.yres)

                inCollision = self.gridmap.inCollision(x, y)

            particles[:, i] = np.array([[x, y, theta]])

        return particles

    def sampleParticlesGaussian(self, x0, y0, sigma):
        """
            Samples the set of particles according to a Gaussian distribution
            Orientation are sampled from a uniform distribution

            Args
            ----------
            x0 :           Mean x-position
            y0  :          Mean y-position
                           (See Table 5.3 in Probabilistic Robotics)
            sigma :        Standard deviation of the Gaussian
        """

        (m, n) = self.gridmap.getShape()

        self.particles = np.empty([3, self.numParticles])

        for i in range(self.numParticles):
            inCollision = True
            while inCollision:
                x = np.random.normal(x0, sigma)
                y = np.random.normal(y0, sigma)
                theta = np.random.uniform(-np.pi, np.pi)
                inCollision = self.gridmap.inCollision(x, y)

            self.particles[:, i] = np.array([[x, y, theta]])

        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def getParticle(self, k):
        """
            Returns desired particle (3 x 1 array) and weight

            Args
            ----------
            k :   Index of desired particle

            Returns
            -------
            particle :  The particle having index k
            weight :    The weight of the particle
        """

        if k < self.particles.shape[1]:
            return self.particles[:, k], self.weights[:, k]
        else:
            print('getParticle: Request for k=%d exceeds number of particles (%d)' % (k, self.particles.shape[1]))
            return None, None

    def getNormalizedWeights(self):
        """
            Returns an array of normalized weights

            Returns
            -------
            weights :  An array of normalized weights (numpy.array)
        """

        return self.weights/np.sum(self.weights)

    def getMean(self):
        """
            Returns the mean of the particle filter distribution

            Returns
            -------
            mean :  The mean of the particle filter distribution (numpy.array)
        """

        weights = self.getNormalizedWeights()
        return np.sum(np.tile(weights, (self.particles.shape[0], 1)) * self.particles, axis=1)

    def render(self, ranges, deltat, XGT):
        """
            Visualize filtering strategies

            Args
            ----------
            ranges :   LIDAR ranges (numpy.array)
            deltat :   Step size
            XGT :      Ground-truth pose (numpy.array)
        """

        self.vis.drawParticles(self.particles)
        if XGT is not None:
            self.vis.drawLidar(ranges, self.laser.Angles, XGT[0], XGT[1], XGT[2])
            self.vis.drawGroundTruthPose(XGT[0], XGT[1], XGT[2])
        mean = self.getMean()
        self.vis.drawMeanPose(mean[0], mean[1])
        plt.pause(deltat)

    def prediction(self, u, deltat):
        """
            Implement the proposal step using the motion model based in inputs
            v (forward velocity) and w (angular velocity) for deltat seconds

            This model corresponds to that in Table 5.3 in Probabilistic Robotics

            Args
            ----------
            u :       Two-vector of control inputs (numpy.array)
            deltat :  Step size
        """

        # Implement the algorithm given in Table 5.3
        v = u[0]
        w = u[1]
        N = self.numParticles
        first_var = self.Alpha[0]*abs(v) + self.Alpha[1]*abs(w)
        second_var = self.Alpha[2]*abs(v)+self.Alpha[3]*abs(w)
        third_var = self.Alpha[4]*abs(v) + self.Alpha[5]*abs(w)
        vhat = v + np.random.normal(scale = first_var, size = N)
        what = w + np.random.normal(scale = second_var, size= N)
        gammahat = np.random.normal(scale = third_var, size= N)
        
        # for loop this part over num particles
        for k in range(N):
            (x, y, theta), _ = self.getParticle(k)
            xprime = x - (vhat[k] / what[k] * np.sin(theta))+ (vhat[k] / what[k] *np.sin(theta+what[k] *deltat))
            yprime = y + (vhat[k] / what[k])*np.cos(theta) - (vhat[k] / what[k])* np.cos(theta+what[k]*deltat)
            if not self.gridmap.inCollision(xprime,yprime):
                thetaprime = theta + what[k] * deltat + gammahat[k] * deltat
                thetaprime = self.angleWrap(thetaprime)
                self.particles[:, k] = np.array([xprime, yprime, thetaprime])

    def angleWrap(self, theta):
        """Ensure that a given angle is in the interval (-pi, pi)."""
        while theta < -np.pi:
            theta = theta + 2*np.pi

        while theta > np.pi:
            theta = theta - 2*np.pi

        return theta

    def residual_resample(self):
        indexes = np.zeros(self.numParticles, 'i')
        num_copies = (np.floor(self.numParticles*np.asarray(self.getNormalizedWeights().ravel()))).astype(int)
        k = 0
        for i in range(self.numParticles):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1

        # use multinormal resample on the residual to fill up the rest. This
        # maximizes the variance of the samples
        residual = self.getNormalizedWeights().ravel() - num_copies     # get fractional part
        residual /= sum(residual)           # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
        indexes[k:self.numParticles] = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, self.numParticles - k))
        return indexes

    def resample(self):
        """
            Perform resampling with replacement
        """

        # The np.random.choice function may be useful
        indices = self.residual_resample()
        self.particles = self.particles[:, indices]
        self.weights = (1./self.numParticles) * np.ones((1, self.numParticles))

    def update(self, ranges):
        """
            Implement the measurement update step

            Args
            ----------
            ranges :    Array of LIDAR ranges (numpy.array)
        """

        # for each particle
        # get weight for each particle p(zt | xt m) 
        # where xt came from preiction function
        # Xt = Xt + <xt, wt>
        for k in range(self.numParticles):
            x, weight = self.getParticle(k)
            prob = self.laser.scanProbability(z = ranges, x = x, gridmap = self.gridmap)
            self.weights[:, k] = weight*prob
        
        self.weights = self.getNormalizedWeights()
    
    def run(self, U, Ranges, deltat, X0, XGT, filename):
        """
            The main loop that runs the particle filter

            Args
            ----------
            U :      An array of control inputs, one column per time step (numpy.array)
            Ranges : An array of LIDAR ranges (numpy,array)
            deltat : Duration of each time step
            X0 :     The initial pose (may be None) (numpy.array)
            XGT :    An array of ground-truth poses (may be None) (numpy.array)
        """

        # TODO: Try different sampling strategies (including different values for sigma)
        sampleGaussian = False
        NeffThreshold = self.numParticles/10
        if sampleGaussian and (X0 is not None):
            sigma = 0.5
            self.sampleParticlesGaussian(X0[0, 0], X0[1, 0], sigma)
        else:
            self.sampleParticlesUniform()

        # Iterate over the data
        for k in range(U.shape[1]):
            u = U[:, k]
            ranges = Ranges[:, k]

            # TODO: Your code goes here
            self.prediction(u, deltat)
            self.update(ranges)
            Neff = 1 / np.sum(self.getNormalizedWeights().flatten()**2)
            if Neff <= NeffThreshold:
                self.resample()

            if self.visualize:
                if XGT is None:
                    self.render(ranges, deltat, None)
                else:
                    self.render(ranges, deltat, XGT[:, k])

        plt.savefig(filename, bbox_inches='tight')
