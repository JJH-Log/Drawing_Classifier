#include<iostream>
#include<vector>
using namespace std;

vector<vector<int>> v;
vector<vector<int>> result;

int main() {
	int i, j, N, M;
	cin >> N >> M;

	v.resize(N, vector<int>(M, 0));
	result.resize(N, vector<int>(M, 0));

	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			cin >> v[i][j];
			if ((i != 0) && (j != 0)) result[i][j] += max(result[i - 1][j], result[i][j - 1]);
			if ((i != 0) && (j == 0)) result[i][j] += result[i - 1][j];
			if ((i == 0) && (j != 0)) result[i][j] += result[i][j - 1];
			result[i][j] += v[i][j];
		}
	}
	cout << result[N - 1][M - 1];
	return 0;
}