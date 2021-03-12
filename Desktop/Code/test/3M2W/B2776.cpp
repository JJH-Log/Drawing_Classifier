#include <iostream>
#include <vector>
#include <algorithm>
#pragma warning(disable:4996) //scanf 를 사용하기 위한 코드
//scanf, printf 안쓰면 런타임 에러남
using namespace std;

vector<int> v;

int main() {
	int i, j, N, M;
	int tcase, temp;
	int l, r, mid;
	bool flag;
	cin >> tcase;
	for (i = 0; i < tcase; i++) {
		scanf("%d", &N);
		v.resize(N, -1);
		for (j = 0; j < N; j++) scanf("%d", &v[j]);
		sort(v.begin(), v.end());
		scanf("%d", &M);
		for (j = 0; j < M; j++) {
			scanf("%d", &temp);
			l = 0;
			r = N - 1;
			flag = false;
			while ((l <= r) && (!flag)) {
				mid = (l + r) / 2;
				if (temp > v[mid]) l = mid + 1;
				if (temp < v[mid]) r = mid - 1;
				if (temp == v[mid]) flag = true;
			}
			if (flag) printf("%d\n", 1);
			if (!flag) printf("%d\n", 0);
		}
	}

	return 0;
}
