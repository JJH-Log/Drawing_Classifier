#include<iostream>
#include<vector>
using namespace std;

vector<vector<int>> v;

int main() {
	int i, j, N, M;
	int result;
	cin >> N >> M;
	v.resize(N, vector<int>(M, 0));
	result = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			cin >> v[i][j];
			if (j != 0) result += abs(v[i][j] - v[i][j-1]);
		}
		result = result + v[i][0] + v[i][M - 1];
	}
	for (j = 0; j < M; j++) {
		for (i = 0; i < N; i++) {
			if (i != 0) result += abs(v[i][j] - v[i - 1][j]);
		}
		result = result + v[0][j] + v[N - 1][j];
	}
	result += 2 * M * N;
	cout << result;
	return 0;
}