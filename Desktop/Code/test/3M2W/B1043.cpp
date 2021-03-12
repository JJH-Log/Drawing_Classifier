#include<iostream>
#include<vector>
using namespace std;

vector<vector<int>> v;
vector<int> man;
#define TRUTH 99

int tie(int n1, int n2) { //man에 있는 모든 n1을 n2로 변경
	int i;
	for (i = 0; i < man.size(); i++) {
		if (man[i] == n1) man[i] = n2;
	}
	return 0;
}

int main() {
	int i, j, N, M, num, temp, result;
	bool know;
	cin >> N >> M;
	man.resize(N + 1, 0);
	v.resize(M);

	cin >> num;
	for (i = 0; i < num; i++) {
		cin >> temp;
		man[temp] = TRUTH;
	}

	for (i = 0; i < M; i++) {
		cin >> num;
		v[i].resize(num, 0);
		temp = i+1;
		for (j = 0; j < num; j++) {
			cin >> v[i][j];
			temp = max(temp, man[v[i][j]]);
		}
		for (j = 0; j < num; j++) {
			if (man[v[i][j]] == 0) man[v[i][j]] = temp;
			else tie(man[v[i][j]], temp);
		}
	}

	result = 0;
	for (i = 0; i < M; i++) {
		know = false;
		for (j = 0; j < v[i].size(); j++) {
			if (man[v[i][j]] == TRUTH) know = true;
		}
		if (!know) result++;
	}

	cout << result;
	return 0;
}