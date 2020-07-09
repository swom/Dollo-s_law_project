#include "dollo.hpp"


namespace dollo {


	//take the indexes of ten best intakes
	std::vector<int> Sort_N_Best(int n, std::vector<int> values_vector)
	{
		std::vector<int> vx;
		vx.resize(values_vector.size());
		for (int i = 0; i < values_vector.size(); ++i) vx[i] = i;
		std::sort(vx.begin(), vx.end(), Comp(values_vector));
		return vx;
	}

	Param::Param(std::filesystem::path parameters_path, std::streampos & bookmark)

	{
		

		std::ifstream parameters(parameters_path);
		parameters.seekg(bookmark);
		if (parameters.is_open()) {

			parameters >> N;
			parameters >> G;
			parameters >> samples;
			parameters >> lambda;
			parameters >> c;
			parameters >> mu;
			parameters >> mu_step;
			parameters >> Fa_a;
			parameters >> Fa_b;
			parameters >> N_a;
			parameters >> N_b;
		}
		else {
			std::cout << "Param not constructed";
			abort();
		}
		bookmark = parameters.tellg();//keep track of where we are reading in the parameter file
		parameters.close();


	}
	;
	Param::Param(std::filesystem::path parameters_path, std::streampos & bookmark, bool & move_bookmark) 
	{


		std::ifstream parameters(parameters_path);
		parameters.seekg(bookmark);
		if (parameters.is_open()) {

			parameters >> N;
			parameters >> G;
			parameters >> samples;
			parameters >> lambda;
			parameters >> c;
			parameters >> mu;
			parameters >> mu_step;
			parameters >> Fa_a;
			parameters >> Fa_b;
			parameters >> N_a;
			parameters >> N_b;
		}
		else {
			std::cout << "Param not constructed";
			abort();
		}

		if(move_bookmark)
		{
			bookmark = parameters.tellg();//keep track of where we are reading in the parameter file
			move_bookmark = 0;
		}
		parameters.close();


	}
	;

	Distributions::Distributions(const Param& param)
	{

		mu = param.mu;
		mu_step = param.mu_step;
		Fa_a = param.Fa_a;
		Fa_b = param.Fa_b;
		N_a = param.N_a;
		N_b = param.N_b;
		
	}



	void access_fitness(Param& param, PopVectors& pop_vec, const Distributions& distributions, rndutils::default_engine & reng, int test_iterations =1)
	{
		float min_intake = std::numeric_limits<float>::max();  
		auto fatty_acid_dist = distributions.fatty_acid_dist();
		auto nutrient_dist = distributions.nutrient_dist();

		for (int i = 0; i < static_cast<int>(param.N); ++i) {
			pop_vec.intake_[i] = 0.f;
			pop_vec.correct_guess_[i] = 0;
			pop_vec.optimal_intake_[i] = 0;

			for (int s = 0; s < param.samples*test_iterations; ++s) {
				const auto E_N = nutrient_dist(reng);
				const auto E_Fa = fatty_acid_dist(reng);                     
				const auto E1 = (1.f - param.lambda) * E_Fa + E_N - param.c; 
				const auto y = pop_vec.pop_[i](E_Fa, E_N)[0];    // output of the GRN-> {0,1}
				const auto E = (1.f - y) * E_Fa + y * E1;
				pop_vec.intake_[i] += E;
				pop_vec.correct_guess_[i] += (y == 0.f) == (E_Fa > E1);
				pop_vec.optimal_intake_[i] += E_Fa*(E_Fa > E1) + E1 * (E1 > E_Fa);
			}
			min_intake = std::min(pop_vec.intake_[i], min_intake);
		}
		// transform intakes into discrete fitness distribution
		pop_vec.fitness_dist_.mutate_transform(pop_vec.intake_.cbegin(), pop_vec.intake_.cend(), [=](float x) { return x - min_intake; });
	}


	void reproduce(Param& param, PopVectors& pop_vec, const Distributions& distributions, rndutils::default_engine & reng)
	{
		// sample parents from fitness distribution and mutate offspring on the fly
		//auto reng = rndutils::make_random_engine<>(param.seed);
		auto mu_dist = distributions.mu_dist();
		auto mu_step_dist = distributions.mu_step_dist();
		for (size_t i = 0; i < param.N; ++i) {
			pop_vec.tmp_[i] = pop_vec.pop_[pop_vec.fitness_dist_(reng)];
			for (auto& w : pop_vec.tmp_[i]) {
				if (mu_dist(reng)) {
					w += mu_step_dist(reng);
				}
			}
		}
		pop_vec.pop_.swap(pop_vec.tmp_);
	}

	

	Bootcamp::Bootcamp( Param param,
			rndutils::default_engine * reng_,	
			Observer<Bootcamp>* observer)
		
	{
		reng = reng_;
		param_ = param;
		distributions_ = Distributions(param_);
		pop_vec_.pop_.resize(param_.N, GRN{ 0 });
		pop_vec_.tmp_.resize(param_.N, GRN{ 0 });
		pop_vec_.intake_.resize(param_.N,0.f);
		pop_vec_.optimal_intake_.resize(param_.N, 0.f);
		pop_vec_.correct_guess_.resize(param_.N,0);
		observer_=observer;
		auto reng = rndutils::make_random_engine<>(param_.seed);
		auto w_dist = distributions_.mu_step_dist();
		for (auto& ind : pop_vec_.pop_) {
			for (auto& w : ind) {
				w = w_dist(reng);
			}
		}
		notify(Event::Initialized);
	}


	void Bootcamp::run()
	{
	  for (size_t g = 0; g < param_.G; ++g) {
	   access_fitness(param_,pop_vec_,distributions_,*reng);
	    notify(Event::FitnessAccessed);
	    reproduce(param_, pop_vec_, distributions_,*reng);
	    notify(Event::Reproduced);
	  }
		access_fitness(param_, pop_vec_, distributions_,*reng, 100);
	  notify(Event::Finished);
	}

	void Bootcamp::notify(Event event) const
	{
		if (observer_) observer_->notify(event, *this);
	}


	Phase2::Phase2(int seeding_individual, 
			int timer_test, int test_iterations,
			Param param, Bootcamp & bootpop,
			rndutils::default_engine * reng_,
			Observer<Phase2>* observer):
   
		
		test_timer_(timer_test),
		test_iterations_ (test_iterations)

	{
		reng = reng_;
		param_ = param;
		distributions_ = Distributions(param_);

		pop_vec_.pop_=bootpop.pop_vec_.pop_;

		pop_vec_.tmp_.resize(param_.N, GRN{ 0 });
		pop_vec_.intake_.resize(param_.N,0.f);
		pop_vec_.optimal_intake_.resize(param_.N, 0.f);
		pop_vec_.correct_guess_.resize(param_.N,0);
		observer_ = observer;
		bootcamp_address_ = &bootpop;
		notify(Event::Initialized);

	}

	void Phase2::run()
	{
		for (size_t g = 0; g < param_.G; ++g) {
			access_fitness(param_, pop_vec_, distributions_, *reng);
			notify(Event::FitnessAccessed);
			reproduce(param_, pop_vec_, distributions_, *reng);
			notify(Event::Reproduced);
			if (g%test_timer_==0) {
				access_fitness(param_, pop_vec_, bootcamp_address_->distributions(),*reng, test_iterations_);
				notify(Event::Testing);
			}
		}
		notify(Event::Finished);
	}

	void Phase2::notify(Event event) const
	{
		if (observer_) observer_->notify(event, *this);
	}



	Phase3::Phase3(
		int seeding_individual,
			int timer_test,
			int test_iterations,
			float probability_of_shift,
			size_t shift_duration,
		const Param &param, 
			Bootcamp& bootpop,
		rndutils::default_engine * reng_, Observer<Phase3>* observer)

	{
		reng = reng_;
		test_timer_=timer_test;
		test_iterations_=test_iterations;
		probability_of_shift_ = probability_of_shift;
		shift_duration_ = shift_duration;
		param_ = param;
		distributions_ = Distributions(param_);

		pop_vec_.pop_ = bootpop.pop_vec_.pop_;

		pop_vec_.tmp_.resize(param_.N, GRN{ 0 });
		pop_vec_.intake_.resize(param_.N, 0.f);
		pop_vec_.optimal_intake_.resize(param_.N, 0.f);
		pop_vec_.correct_guess_.resize(param_.N, 0);
		observer_ = observer;
		bootcamp_address_ = &bootpop;
		notify(Event::Initialized);
	}

	void Phase3::run()
	{
		//auto reng = rndutils::make_random_engine<>(param_.seed);
		auto unif_dist = distributions_.unif_dist();

		for (size_t g = 0; g < param_.G; ++g) {
		
			if ( unif_dist(*reng) < probability_of_shift_) {

				for (size_t shift_generations = 0; shift_generations < shift_duration_;shift_generations++,g++) {
					access_fitness(param_, pop_vec_, bootcamp_address_->distributions(),*reng);
					notify(Event::FitnessAccessed);

					reproduce(param_, pop_vec_, distributions_, *reng);
					notify(Event::Reproduced);

					if (g%test_timer_ == 0) {
						access_fitness(param_, pop_vec_, bootcamp_address_->distributions(), *reng, test_iterations_);
						notify(Event::Testing);
					}
				}
				g--;
			}
			else
			{
				access_fitness(param_, pop_vec_, distributions_, *reng);
				notify(Event::FitnessAccessed);

				reproduce(param_, pop_vec_, distributions_, *reng);
				notify(Event::Reproduced);

				if (g%test_timer_ == 0) {
					access_fitness(param_, pop_vec_, bootcamp_address_->distributions(), *reng, test_iterations_);
					notify(Event::Testing);
				}
			}	

		}
		notify(Event::Finished);
	}

	void Phase3::notify(Event event) const
	{
		if (observer_) observer_->notify(event, *this);
	}


}
