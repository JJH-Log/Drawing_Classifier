#include<iostream>
#include<vector>
using namespace std;

vector<int> v;

int main() {
	int i, j, N, M;
	int result;
	int temp, count;
	temp = 0;
	result = 0;
	cin >> N >> M;

	v.resize(M, 0);

	for (i = 0; i < M; i++) cin >> v[i];

	for (i = 0; i < M; i++) {
		count = 0;
		for (j = 0; j < M; j++) {
			if (v[i] <= v[j]) count++;
		}
		if (count > N) count = N;
		if (temp < count * v[i]) {
			result = v[i];
			temp = count * v[i];
		}
	}
	cout << result << " " << temp;
}