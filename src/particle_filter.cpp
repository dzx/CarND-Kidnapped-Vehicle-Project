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
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for(int i = 0; i != num_particles; i++){
    Particle new_part;
    new_part.id = i;
    new_part.x = dist_x(gen);
    new_part.y = dist_y(gen);
    new_part.theta = dist_theta(gen);
    new_part.weight = 1.;
    particles.push_back(new_part);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0., std_pos[0]);
  normal_distribution<double> dist_y(0., std_pos[1]);
  normal_distribution<double> dist_theta(0., std_pos[2]);
  for(int i = 0; i != particles.size(); i++){
    Particle& current = particles[i];
    if(yaw_rate != 0.){
      long double vel_rate = velocity / yaw_rate;
      long double n_yaw = current.theta + yaw_rate*delta_t;
      current.x += (vel_rate * (sin((long double)n_yaw) - sin((long double)current.theta)));
      current.y += (vel_rate * (cos((long double)current.theta) - cos((long double)n_yaw)));
      current.theta = n_yaw;
    }else{
      current.x += velocity * cos((long double)current.theta) * delta_t;
      current.y += velocity * sin((long double)current.theta) * delta_t;
    }
    current.x += dist_x(gen);
    current.y += dist_y(gen);
    current.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs>& observations,
                                     std::vector<int>& associations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  if(!associations.empty()) associations.clear();
  for(int i = 0; i != observations.size(); i++){
    int association = predicted.size();
    double min_dist = 0;
    for(int j = 0; j != predicted.size(); j++){
      double dist = sqrt(pow(predicted[j].x_f - observations[i].x, 2) + pow(predicted[j].y_f - observations[i].y, 2));
      if(dist < min_dist || association == predicted.size()){
        min_dist = dist;
        association = j;
      }
    }
    associations.push_back(association);
//    std::cout << "obsx:" << observations[i].x << " obsY: " << observations[i].y << " lm:" << association <<
//        " lmX:" << predicted[association].x_f << " lmY:" << predicted[association].y_f << std::endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
  const double pi = std::acos(-1); //
  for(std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); it++){
    Particle& current = *it;
    double cos_t = cos(current.theta);
    double sin_t = sin(current.theta);
    std::vector<LandmarkObs> trn_obs;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for(std::vector<LandmarkObs>::iterator obs_it = observations.begin(); obs_it != observations.end(); obs_it++){
      LandmarkObs trn_observation;
      trn_observation.id = obs_it->id;
      trn_observation.x = current.x + cos_t * obs_it->x - sin_t * obs_it->y;
      trn_observation.y = current.y + sin_t * obs_it->x + cos_t * obs_it-> y;
      trn_obs.push_back(trn_observation);
      sense_x.push_back(trn_observation.x);
      sense_y.push_back(trn_observation.y);
    }
    std::vector<int> associations;
    associations.clear();
    dataAssociation(map_landmarks.landmark_list, trn_obs, associations);
//    current = SetAssociations(current, associations, sense_x, sense_y);
    sense_x.clear();
    sense_y.clear();
    current.weight = 1.;
    for(size_t i = 0; i != trn_obs.size(); i++){
      LandmarkObs obs = trn_obs[i];
      Map::single_landmark_s pred = map_landmarks.landmark_list[associations[i]];
      double diff_x = obs.x - pred.x_f;
      double diff_y = obs.y - pred.y_f;
      double inv_norm_factor = 2. * pi * std_landmark[0] * std_landmark[1];
      double neg_exp_factor = diff_x * diff_x / (2. * std_landmark[0] * std_landmark[0]) +
          diff_y * diff_y / (2. * std_landmark[1] * std_landmark[1]);
      double part_prob = exp(-neg_exp_factor) / inv_norm_factor;
      current.weight *= part_prob;
//      std::cout << "dx:" << diff_x << "dy:" << diff_y << "nf:" << inv_norm_factor << "ef:" <<
//          neg_exp_factor << "p_prob:" << part_prob << std::endl;
    }
    trn_obs.clear();
//    std::cout << "Particle: " << current.id << " Weight: " << current.weight << std::endl;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  double max_weight = 0;
  for(std::vector<Particle>::iterator it = particles.begin(); it != particles.end(); it++){
    if(it->weight > max_weight) max_weight = it->weight;
  }
  double beta = 0.;
  std::vector<Particle> resampled;
  default_random_engine gen;
  uniform_int_distribution<size_t> dist_init(0, particles.size()-1);
  uniform_real_distribution<double> dist_beta(0, max_weight * 2.);
  int pick = dist_init(gen);
  for(size_t i = 0; i != particles.size(); i++){
    beta += dist_beta(gen);
    while(beta > particles[pick].weight){
      beta -= particles[pick++].weight;
      if(pick == particles.size()) pick = 0;
    }
    resampled.push_back(particles[pick]);
//    std::cout << "Picked:" << pick << " weight:" << particles[pick].weight << std::endl;
  }
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
