#ifndef DOLLO_DOLLO_HPP_INCLUDED
#define DOLLO_DOLLO_HPP_INCLUDED

#include <iostream>
#include <fstream>
#include<filesystem>
#include <stdexcept>
#include <vector>
#include <array>
#include <algorithm>
#include "rndutils.hpp"
#include "ann.hpp"


namespace dollo{

  using namespace ann;


  using GRN = ann::Network<float,

	  
	  Layer<Neuron<2, activation::sgn::unipolar>, 1>
  >;
	

	std::vector<int> Sort_N_Best(int n, std::vector<int> values_vector);
	struct Comp {
		Comp(const std::vector<int>& v) : _v(v) {}
		bool operator ()(int a, int b) { return _v[a] > _v[b]; }
		const std::vector<int>& _v;
	};


	

	struct Param
	{

		Param() {};
		//Param(int seed);
		//Param(float fat_min, float fat_max, size_t N, size_t G, int seed );
		//Param(size_t N, size_t G, int samples, float lambda, float c, float mu, float mu_step, float  Fa_a, float Fa_b, float N_a, float N_b, int seed);
		Param(std::filesystem::path parameters_path, std::streampos & bookmark);

		//this is just an overlaod where the bookmar of the stream file is not changed to build multiple models with the same parameters
		Param(std::filesystem::path parameters_path, std::streampos & bookmark, bool & move_bookmark);

		int seed;
		size_t N;        // number of individuals
		size_t G;           // generations
		int samples;         // samples per generations

		float lambda;      // penalty fatty acids intake in pathway
		float c;            // static cost of pathway

		double mu;        // mutation prob. per weight
		float mu_step;     // mutation step parameter, e.g. stddev

		float Fa_a;       // 1st parameter fatty acid distribution
		float Fa_b;       // 2nd parameter fatty acid distribution
		float N_a;        // 1st parameter nutrient distribution
		float N_b;
		std::streampos pos;
	};

	struct Distributions
	{
		explicit Distributions(const Param& param);
	
		double mu;      // mutation prob. per weight
		float mu_step ;     // mutation step parameter, e.g. stddev

		float Fa_a ;       // 1st parameter fatty acid distribution
		float Fa_b ;       // 2nd parameter fatty acid distribution
		float N_a ;        // 1st parameter nutrient distribution
		float N_b ;        // 2nd parameter nurtient distribution

		auto unif_dist() const { return std::uniform_real_distribution<float>(0.0000f, 1.0000f); }
		auto fatty_acid_dist() const { return std::uniform_real_distribution<float>(Fa_a, Fa_b); }
		auto nutrient_dist() const { return std::uniform_real_distribution<float>(N_a, N_b); }
		auto mu_dist() const { return std::bernoulli_distribution(mu); }                     // mutation prob. per weight distribution
		auto mu_step_dist() const { return std::normal_distribution<float>(0.f, mu_step); }  // mutation step distribution
	};

	struct PopVectors
	{
		std::vector<GRN> pop_;
		std::vector<GRN> tmp_;             // ping-pong buffer offsprings/parents
		std::vector<float> intake_;        // energy intake per ind over one generation
		std::vector<float> optimal_intake_;        // energy intake per ind over one generation
		std::vector<int> correct_guess_;   // number of correct decission per ind over one generation
		using fitness_dist_t = rndutils::mutable_discrete_distribution<>;
		fitness_dist_t fitness_dist_;

	};

  template <typename Observed>
  class Observer
  {
  public:
    using Event = typename Observed::Event;
    virtual ~Observer() {}
    virtual void notify(Event, const Observed &) {};
  };

	class Simulation
	{

	public:

		enum class Event {
			FitnessAccessed,
			Reproduced,
			Initialized,
			Shift,
			Testing,
			Finished
		};

		// grant read-only access
		const auto& pop() const noexcept { return pop_vec_.pop_; }
		const auto& tmp() const noexcept { return pop_vec_.tmp_; }
		const auto& param() const noexcept { return param_; }
		const auto& intake() const noexcept { return pop_vec_.intake_; }
		const auto& optimal_intake() const noexcept { return pop_vec_.optimal_intake_; }
		const auto& correct_guess() const noexcept { return pop_vec_.correct_guess_; }
		const auto& fitness_dist() const noexcept { return pop_vec_.fitness_dist_; }
		const auto& distributions() const noexcept { return distributions_; }
		const auto& pop_vec() const noexcept { return pop_vec_; }
		PopVectors pop_vec_;

	protected:
		
		//int test_timer_, test_iterations_;


		Param param_;
		Distributions distributions_ = Distributions(param_);
		rndutils::default_engine *reng;

	};

  class Bootcamp: public Simulation
  {
  
  public:
    explicit Bootcamp( Param param_,rndutils::default_engine * reng_, Observer<Bootcamp>* observer = nullptr);
    void run();

		const auto& fat_dist() const noexcept { return distributions_.fatty_acid_dist(); }
		const auto& nut_dist() const noexcept { return distributions_.nutrient_dist(); }

		void notify(Event event) const;
		

	protected: 
		Observer<Bootcamp> * observer_;

  };

	class Phase2 : public Simulation
	{

	public:
		explicit Phase2(int seeding_individual, int timer_test, int test_iterations, Param param, Bootcamp& bootpop, rndutils::default_engine * reng_, Observer<Phase2>* observer = nullptr);

		void run();
		void notify(Event event) const;
		int test_timer_, test_iterations_;


	protected:

		Observer<Phase2>* observer_;
		Bootcamp * bootcamp_address_;
	};

	class Phase3 : public Simulation
	{

	public:
		explicit Phase3( int seeding_individual, int timer_test, int test_iterations, float probability_of_shift, size_t shift_duration,const Param &param, Bootcamp& bootpop, rndutils::default_engine * reng_, Observer<Phase3>* observer = nullptr);

		void run();
		void notify(Event event) const;
		int test_timer_, test_iterations_;


	protected:
		float probability_of_shift_;
		size_t shift_duration_;
		Observer<Phase3>* observer_;
		Bootcamp * bootcamp_address_;

	};
}

#endif
