#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

bool sortinrev(const pair<int, int> &a, const pair<int, int> &b) {
	if (a.first > b.first) {
		return true;
	}
	else if (a.first < b.first) {
		return false;
	}
	else { // a.first == b.first
		if (a.second < b.second) {
			return true;
		}
		else {
			return false;
		}
	}
}


int Q2(int k) {

	vector<int> v, first_occur_time;
	vector<pair<int, int> > frequency;

	for (int i = 1; i < 2 * k + 1; ++i) {
		v.push_back(i);
		first_occur_time.push_back(0);
		frequency.push_back(make_pair(0, i));
	};

	vector<int> new_v = v;

	for (int m = 1; m < k + 1; ++m) {
		for (int i = 0; i < m; ++i) {
			new_v[2 * i] = v[m + i];
			new_v[2 * i + 1] = v[i];
		}
		v = new_v;
		frequency[v[0] - 1].first += 1;
		if (first_occur_time[v[0] - 1] == 0) {
			first_occur_time[v[0] - 1] = m;
		};

	};

	for (int m = k; m >0; --m) {
		for (int i = 0; i < m; ++i) {
			new_v[2 * i] = v[m + i];
			new_v[2 * i + 1] = v[i];
		}
		v = new_v;
		frequency[v[0] - 1].first += 1;
		if (first_occur_time[v[0] - 1] == 0) {
			first_occur_time[v[0] - 1] = 2 * k - m + 1;
		};
	};

	int top = v[0];
	int top_frequency;
	top_frequency = frequency[top - 1].first;
	sort(frequency.begin(), frequency.end(), sortinrev);


	cout << "The answer to Q2(a): " << " ";
	for (int i = 0; i < 3; ++i) cout << v[i] << " ";
	cout << endl;
	cout << "The answer to Q2(b): " << " ";
	for (int i = 0; i < 3; i++) {
		cout << frequency[i].second << "  ";
	}
	cout << " [" << " ";
	for (int i = 0; i < 3; i++) {
		cout << frequency[i].first<< "  ";
	}
	cout << "times]" << endl;
	cout << "The answer to Q2(c):  " << first_occur_time[top - 1] << endl;
	cout << "The answer to Q2(d):  " << top_frequency << endl;

	return 0;

}

int main() {
	int k;
	cout << "Choose your k size: ";
	cin >> k;
	Q2(k);
	cout << "press 0 to exit";
	cin >> k;
}

