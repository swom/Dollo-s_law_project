
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include "dollo.hpp"
#include "rndutils.hpp"
#include <sstream>
#include <string>

template<class T>
std::string toString(const T& value) {
	std::ostringstream os;
	os << value;
	return os.str();
}

//overload to print on console
void printing_Statistics(const dollo::Param& param, const dollo::PopVectors & pop_vec, const dollo::Distributions& distributions, int & g_) {
	const auto& p = param;
	std::cout << std::setprecision(3);
	std::cout << g_++ << "  ";
	const auto n_intake = std::accumulate(pop_vec.intake_.cbegin(), pop_vec.intake_.cend(), 0.f);
	const auto n_guess = std::accumulate(pop_vec.correct_guess_.cbegin(), pop_vec.correct_guess_.cend(), 0);
	const auto n_optimal_intake = std::accumulate(pop_vec.optimal_intake_.cbegin(), pop_vec.optimal_intake_.cend(), 0.f);
	std::cout << "  " << n_intake / p.N << "  ";
	std::cout << static_cast<double>(n_guess) / (p.N * p.samples)<<"  ";
	std::cout << n_optimal_intake / p.N << "  ";

	// let's do the same for the top Q (if only to introduce nth_element)
	const auto Q = p.N / 10;
	auto idx = std::vector<int>(p.N);
	std::iota(idx.begin(), idx.end(), 0); //fills the idx vector with 0s, why not simply initiate it in the previous line?
	auto first = idx.begin();
	auto nth = idx.begin() + Q;
	std::nth_element(first, nth, idx.end(), [&intake = pop_vec.intake_](int i, int j) {
		return intake[i] > intake[j];    // descending sorting
	});
	const auto q_intake = std::accumulate(first, nth, 0.f, [&intake = pop_vec.intake_](float s, int i) {
		return s + intake[i];
	});
	const auto q_optimal_intake = std::accumulate(first, nth, 0.f, [&optimal_intake = pop_vec.optimal_intake_](float s, int i) {
		return s + optimal_intake[i];
	});
	const auto q_guess = std::accumulate(first, nth, 0, [&correct = pop_vec.correct_guess_](int s, int i) {
		return s + correct[i];
	});
	std::cout << "  [" << q_intake / Q << "  ";
	std::cout << static_cast<double>(q_guess) / (Q * p.samples) << "  ";
	std::cout << q_optimal_intake/Q <<"]\n";
}

//overload to print on file
void printing_Statistics(const dollo::Param& param, const dollo::PopVectors & pop_vec, const dollo::Distributions& distributions, int & g_, std::ofstream & file, int num_of_iterations=1) {
	const auto& p = param;
	file << std::setprecision(3);
	file << g_++ << " ";

	//calculate mean and variance of all weights
		// calculate mean and variance of weights
	std::array<float, dollo::GRN::state_size> mean;
	mean.fill(0.f);
	for (size_t i = 0; i < param.N; ++i) {
		const auto ann = pop_vec.pop_[i].cbegin();
		for (size_t w = 0; w < dollo::GRN::state_size; ++w) {
			mean[w] += ann[w];
		}
	}
	for (auto& m : mean) {
		m *= (1.f / static_cast<float>(param.N));
	}

	std::array<float, dollo::GRN::state_size> var;
	var.fill(0.f);
	for (size_t i = 0; i < param.N; ++i) {
		const auto ann = pop_vec.pop_[i].cbegin();
		for (size_t w = 0; w < dollo::GRN::state_size; ++w) {
			var[w] += (mean[w] - ann[w]) * (mean[w] - ann[w]);
		}
	}
	for (auto& v : var) {
		v *= (1.f / static_cast<float>(param.N));
	}


	const auto n_intake = std::accumulate(pop_vec.intake_.cbegin(), pop_vec.intake_.cend(), 0.f);
	const auto n_optimal_intake = std::accumulate(pop_vec.optimal_intake_.cbegin(), pop_vec.optimal_intake_.cend(), 0.f);
	const auto n_guess = std::accumulate(pop_vec.correct_guess_.cbegin(), pop_vec.correct_guess_.cend(), 0);

	file << " " << static_cast<double> (n_intake) / (p.N*num_of_iterations) << " ";
	file << static_cast<double>(n_guess) / (p.N * p.samples*num_of_iterations)<<" ";
	file << n_optimal_intake / (p.N*num_of_iterations) << " ";

	//print means of weigths
	for (size_t i = 0; i < mean.size(); i++)
	{
		file << mean[i] << " ";

	}

	//print sd of weigths
	for (size_t i = 0; i < var.size(); i++)
	{
		file << var[i] << " ";

	}

	// let's do the same for the top Q (if only to introduce nth_element)
	const auto Q = p.N / 10;
	auto idx = std::vector<int>(p.N);
	std::iota(idx.begin(), idx.end(), 0); //fills the idx vector with 0s, why not simply initiate it in the previous line?
	auto first = idx.begin();
	auto nth = idx.begin() + Q;
	std::nth_element(first, nth, idx.end(), [&intake = pop_vec.intake_](int i, int j) {
		return intake[i] > intake[j];    // descending sorting
	});
	const auto q_intake = std::accumulate(first, nth, 0.f, [&intake = pop_vec.intake_](float s, int i) {
		return s + intake[i];
	});
	const auto q_optimal_intake = std::accumulate(first, nth, 0.f, [&optimal_intake = pop_vec.optimal_intake_](float s, int i) {
		return s + optimal_intake[i];
	});
	const auto q_guess = std::accumulate(first, nth, 0, [&correct = pop_vec.correct_guess_](int s, int i) {
		return s + correct[i];
	});
	file << /*"  [" <<*/ q_intake / (Q * num_of_iterations )<<" ";
	file << static_cast<double>(q_guess) / (Q * p.samples*num_of_iterations) /*<< "]"*/<<"  ";
	file << q_optimal_intake/ (Q * num_of_iterations) <<"\n";
}

// Count number of lines in the parameter file to adjust the number of loops accordingly
size_t count_set_of_parameters(std::filesystem::path param_path) {
	std::ifstream parameter(param_path);
	std::string s;
	size_t sTotal = 0;
	while (!parameter.eof()) {
		std::getline(parameter, s);
		std::getline(parameter, s); //we get two lines because for every simulation 
																//there are two lines one for bootcamp and one for phase2

		sTotal++;
	}
	parameter.close();
	return sTotal;
}


class SuperFancyObserverBootCamp : public dollo::Observer<dollo::Bootcamp>
{
public:
	SuperFancyObserverBootCamp(std::filesystem::path path) :
	bootcamp_os_(path)
	{}

	void notify(Event event, const dollo::Bootcamp& model) override
	{
		switch (event) {
		case Event::FitnessAccessed: {
			printing_Statistics(model.param(), model.pop_vec(), model.distributions(), g_ , bootcamp_os_);
			break;
		}
			case Event::Finished: {
				std::cout << "regards\n";
				bootcamp_os_.close();
				break;
		
			}
		}
	}

private:
	int g_ = 0;   // generation counter
	std::ofstream bootcamp_os_; // print file

};


class SuperFancyObserverPhase2 : public dollo::Observer<dollo::Phase2>
{
public:
	SuperFancyObserverPhase2(std::filesystem::path path_normal, std::filesystem::path path_test) :
		phase2_test_os_(path_test),
		phase2_os_(path_normal) 
	{}

  void notify(dollo::Phase2::Event event, const dollo::Phase2& model) override
  {
    switch (event) {
			case Event::FitnessAccessed: {
				printing_Statistics(model.param(), model.pop_vec(), model.distributions(), g_ , phase2_os_);
				break;
			}
			case Event::Testing: 
			{				
			//std::cout << "TESTING  ";
			printing_Statistics(model.param(), model.pop_vec(), model.distributions(), t_ ,phase2_test_os_, model.test_iterations_);

				break;
			 }
   
			case Event::Finished: {
				std::cout << "regards\n";
				phase2_test_os_.close();
				phase2_os_.close();
				break;

			}

    }
  }

private:
  int g_ = 0;   // generation counter
	int t_ = 0;  // test counter
	std::ofstream phase2_os_; // print file for normal evolution
	std::ofstream phase2_test_os_; // print file for test

};

class SuperFancyObserverPhase3 : public dollo::Observer<dollo::Phase3>
{
public:
	SuperFancyObserverPhase3(std::filesystem::path path_normal, std::filesystem::path path_test) :
		phase3_os_(path_normal),
		phase3_test_os_(path_test)

	{}

	void notify(dollo::Phase3::Event event, const dollo::Phase3& model) override
	{
		switch (event) {
		case Event::FitnessAccessed: {
			printing_Statistics(model.param(), model.pop_vec(), model.distributions(), g_, phase3_os_);
			break;
		}

		case Event::Testing:
		{
			printing_Statistics(model.param(), model.pop_vec(), model.distributions(), t_, phase3_test_os_, model.test_iterations_);

			break;
		}

		case Event::Finished: {
			std::cout << "regards\n";
			phase3_os_.close();
			break;

		}

		}
	}

private:
	int g_ = 0;   // generation counter
	int t_ = 0;  // test counter

	std::ofstream phase3_os_; // print file for normal evolution
	std::ofstream phase3_test_os_; // print file for test

};

int main()
{
	int number_of_seeding_individuals = 10;
	int timer_test = 10;
	int test_iterations_in_phase2 = 100;

	size_t shift_duration = 3;




	
	for (int seed = 0; seed < 100; seed++) {

		rndutils::default_engine reng = rndutils::make_random_engine<>(seed);


		//assigning output folder and files
		std::string directory_name = "Seed_" + std::to_string(seed)+"_complexGRN";
		std::filesystem::create_directories(directory_name);
		std::filesystem::path print_path = std::filesystem::current_path() / directory_name;
		//setting parameter file
		std::streampos bookmark = 0; //keeps track of where we are reading from the txt file, every seed will be iterated for different parameters
		std::filesystem::path param_path = std::filesystem::current_path() / "Parameters.txt";



		for (size_t i = 0; i < count_set_of_parameters(param_path); i++)
		{
			try {
				auto observer = SuperFancyObserverBootCamp(print_path / "bootcamp.txt");
				auto model = dollo::Bootcamp(
					dollo::Param(param_path, bookmark),
					&reng,
					&observer
				);

				model.run();


				std::vector<int> start_pop_indexes = dollo::Sort_N_Best(number_of_seeding_individuals, model.correct_guess());
				bool move_bookmark = 0;

				
					std::vector<int> p_shifts{ 0,2,3 };
					for (size_t shift_prob = 0; shift_prob < p_shifts.size(); shift_prob++)
					{
						float probability_of_shift = ( 1.f - (p_shifts[shift_prob] == 0) ) / ( powf(10.f,static_cast<float> (p_shifts[shift_prob])));

					

					
					std::string individual_directory = "Pop_1000000Gen";
					std::filesystem::create_directories(print_path / individual_directory);


					auto observer2 = SuperFancyObserverPhase2(print_path / individual_directory / "phase2.txt", print_path / individual_directory / "testPh2.txt");
					auto model2 = dollo::Phase2(
						start_pop_indexes[0],
						timer_test,
						test_iterations_in_phase2,
						dollo::Param(param_path, bookmark, move_bookmark),
						model,
						&reng,
						&observer2
					);

					std::string phase_3_name = "phase3_Shift_Prob_" + std::to_string(probability_of_shift) + ".txt";
					std::string phase_3_test_name = "phase3_Test_Shift_Prob_" + std::to_string(probability_of_shift) + ".txt";

					auto observer3 = SuperFancyObserverPhase3(print_path / individual_directory / phase_3_name, print_path / individual_directory / phase_3_test_name);
					auto model3 = dollo::Phase3(
						start_pop_indexes[0 ],
						timer_test,
						test_iterations_in_phase2,
						probability_of_shift,
						shift_duration,
						model2.param(),
						model,
						&reng,
						&observer3
					);

					model3.run();
				}
				

	
			}
			catch (const std::exception & err) {
				std::cerr << "Exception: " << err.what() << std::endl;
			}
			catch (...) {
				std::cerr << "Unknown exception" << std::endl;
			}
		}
		

	}

  return 0;
}
