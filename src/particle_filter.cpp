/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
#define EPS 0.00001
#define num_particles 100

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	  default_random_engine gen;

	  // normal (Gaussian) distribution for sensor noise
	  normal_distribution<double> dist_x(0, std[0]);
	  normal_distribution<double> dist_y(0, std[1]);
	  normal_distribution<double> dist_theta(0, std[2]);

	  // Initialize all particles to first position
	  for (int i = 0; i < num_particles; i++) {
		Particle Prt;
		Prt.id = i;
		Prt.x = x;
		Prt.y = y;
		Prt.theta = theta;
		Prt.weight = 1.0;

		// Add random Gaussian noise to each particle.
		Prt.x += dist_x(gen);
		Prt.y += dist_y(gen);
		Prt.theta += dist_theta(gen);

		particles.push_back(Prt);
	  }

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	  default_random_engine gen;

	  // normal (Gaussian) distribution for sensor noise
	  normal_distribution<double> dist_x(0, std_pos[0]);
	  normal_distribution<double> dist_y(0, std_pos[1]);
	  normal_distribution<double> dist_theta(0, std_pos[2]);

	  for (int index_part = 0; index_part < num_particles; index_part++) {

	    // predict new state
	    if (fabs(yaw_rate) < EPS) {
	      particles[index_part].x += velocity * delta_t * cos(particles[index_part].theta);
	      particles[index_part].y += velocity * delta_t * sin(particles[index_part].theta);
	    }
	    else {
	      particles[index_part].x += velocity / yaw_rate * (sin(particles[index_part].theta + yaw_rate*delta_t) - sin(particles[index_part].theta));
	      particles[index_part].y += velocity / yaw_rate * (cos(particles[index_part].theta) - cos(particles[index_part].theta + yaw_rate*delta_t));
	      particles[index_part].theta += yaw_rate * delta_t;
	    }

	    // add random Gaussian noise
	    particles[index_part].x += dist_x(gen);
	    particles[index_part].y += dist_y(gen);
	    particles[index_part].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int index_obs = 0; index_obs < observations.size(); index_obs++) {

	    // store the current observations
	    LandmarkObs obs = observations[index_obs];

	    // inititialize the nearest neighbor to the maximum possible
	    double nearest_neighbor = numeric_limits<double>::max();
	    int landmark_id = -1;

	    for (unsigned int index_prd = 0; index_prd < predicted.size(); index_prd++) {
	      // store the current predictions
	      LandmarkObs pred = predicted[index_prd];

	      // calculate the delta between the observed and predicted px and py
	      double current_neighbor = dist(obs.x, obs.y, pred.x, pred.y);

	      // get the nearest landmark next to the observation
	      if (current_neighbor < nearest_neighbor) {
	    	  nearest_neighbor = current_neighbor;
	          landmark_id = pred.id;
	      }
	    }

	    // set the observation id to the nearest landmark id
	    observations[index_obs].id = landmark_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	// for each particle...

	  for (int index_part = 0; index_part < num_particles; index_part++) {

	    // initialize the particle px and py coordinates
	    double x_part = particles[index_part].x;
	    double y_part = particles[index_part].y;
	    double theta_part = particles[index_part].theta;

	    // get all the landmarks within the sensor range of the particle
	    vector<LandmarkObs> landmark_pred;

	    for (unsigned int index_land = 0; index_land < map_landmarks.landmark_list.size(); index_land++) {

	      float x_land = map_landmarks.landmark_list[index_land].x_f;
	      float y_land = map_landmarks.landmark_list[index_land].y_f;
	      int id_land = map_landmarks.landmark_list[index_land].id_i;

	      // consider landmarks within the circular region of the particle
	      if (((x_land - x_part)*(x_land - x_part)+(y_land - y_part)*(y_land - y_part))<= (sensor_range*sensor_range)) {


	    	  landmark_pred.push_back(LandmarkObs{ id_land, x_land, y_land });
	      }
	    }

	    // perform observation measurement transformations from its local car coordinate system
	    // to the map's coordinate system using Homogenous Transformation
	    vector<LandmarkObs> homogenous_obs;
	    for (unsigned int index_obs = 0; index_obs < observations.size(); index_obs++) {

	      double x_homogenous = cos(theta_part)*observations[index_obs].x - sin(theta_part)*observations[index_obs].y + x_part;
	      double y_homogenous = sin(theta_part)*observations[index_obs].x + cos(theta_part)*observations[index_obs].y + y_part;

	      homogenous_obs.push_back(LandmarkObs{ observations[index_obs].id, x_homogenous, y_homogenous});
	    }

	    // identify measurement landmark associations
	    dataAssociation(landmark_pred, homogenous_obs);

	    particles[index_part].weight = 1.0;

	    for (unsigned int index_hom = 0; index_hom < homogenous_obs.size(); index_hom++) {

	      double x_pred_id, y_pred_id;

	      for (unsigned int index_id = 0; index_id < landmark_pred.size(); index_id++) {
	        if (landmark_pred[index_id].id == homogenous_obs[index_hom].id) {
	          x_pred_id = landmark_pred[index_id].x;
	          y_pred_id = landmark_pred[index_id].y;
	        }
	      }

	      // calculate particle weights as the product
	      // of each measurement's Multivariate-Gaussian probability density
	      double x_std = std_landmark[0];
	      double y_std = std_landmark[1];
	      double weight_obs = ( 1/(2*M_PI*x_std*y_std)) * exp( -( pow(x_pred_id-homogenous_obs[index_hom].x,2)/(2*pow(x_std, 2)) + (pow(y_pred_id-homogenous_obs[index_hom].y,2)/(2*pow(y_std, 2))) ) );

	      particles[index_part].weight *= weight_obs;
	  }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	    default_random_engine gen;
	    vector<Particle> particles_resampled;
        vector<double> weights;


		for(int index_weights=0; index_weights<num_particles; ++index_weights){
			weights.push_back(particles[index_weights].weight);
		}

		// Resample particles

		discrete_distribution<int> weights_distr(weights.begin(),weights.end());

		particles_resampled.resize(num_particles);

		for(int index_part=0; index_part<num_particles; ++index_part){

			auto index_gen = weights_distr(gen);
			particles_resampled[index_part] = particles[index_gen];
		}

	particles = particles_resampled;

}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
