#include <stdio.h>
#include <config.h>
#include <complex.h>
#include <cu/cu.h>

#define EPS 1e-7

int six_invert(_Complex double a[6][6]);

TEST(clover_six_invert) {
  _Complex double a[6][6], b[6][6];
  int test = 0;

  // random matrix a  
  a[0][0] = -0.0226172-1.0842742*I; a[0][1] = -0.4641519+0.7071808*I; a[0][2] = -0.0786318+1.4290063*I;
  a[1][0] =  0.2165182+2.6528579*I; a[1][1] =  1.4397192-0.5239191*I; a[1][2] = -0.7269084+0.8157988*I;
  a[2][0] = -0.0628841-0.3470563*I; a[2][1] = -1.0386082-0.2135166*I; a[2][2] = -1.3647777+0.7312646*I;
  a[3][0] = -0.1675412-0.7309873*I; a[3][1] =  0.1120023-1.3983000*I; a[3][2] = -0.1266411+0.4298037*I;
  a[4][0] = -0.2725515+0.1809753*I; a[4][1] = -0.1379395-0.7037811*I; a[4][2] = -0.6896344+0.1783902*I;
  a[5][0] = -1.0980302+0.2763006*I; a[5][1] = -1.8903566-0.3511587*I; a[5][2] =  1.1886761-1.7150829*I;
  
  a[0][3] =  0.5028327+1.1093231*I; a[0][4] =  0.3878236-1.3375976*I; a[0][5] =  0.1203910+2.0495843*I;
  a[1][3] = -0.5099459-0.0617545*I; a[1][4] =  1.6599072-0.1078419*I; a[1][5] =  0.5164999+1.0314383*I;
  a[2][3] = -0.6036081+0.3900738*I; a[2][4] = -0.0447905+0.7071715*I; a[2][5] =  0.6763751+0.4613504*I;
  a[3][3] =  1.0440726+1.4681992*I; a[3][4] = -1.3339747+0.0932149*I; a[3][5] =  0.3268227-0.4352195*I;
  a[4][3] = -0.3226257-0.8897978*I; a[4][4] = -0.2680521+0.1304365*I; a[4][5] = -1.0114200-0.2461815*I;
  a[5][3] = -0.1194779-0.4089390*I; a[5][4] = -0.1003558+1.6537274*I; a[5][5] = -0.6532741+0.5098912*I;
  
  // b = inverse of a
  b[0][0] = -0.24037097+0.14414191*I; b[0][1] = -0.11380668-0.08118723*I; b[0][2] = -0.1589440+0.4350548*I;
  b[1][0] = -0.10475996+0.12442873*I; b[1][1] =  0.10510192+0.23615703*I; b[1][2] = -0.0141379+0.2762152*I;
  b[2][0] = -0.01620610+0.00456679*I; b[2][1] =  0.02483109-0.02776261*I; b[2][2] = -0.1478979-0.0784658*I;
  b[3][0] =  0.09209149+0.00787285*I; b[3][1] =  0.01995269+0.00092068*I; b[3][2] = -0.2347910+0.1687461*I;
  b[4][0] =  0.21497592+0.31304060*I; b[4][1] =  0.24420948-0.01908121*I; b[4][2] =  0.3385191-0.2141792*I;
  b[5][0] = -0.01061067-0.16808488*I; b[5][1] =  0.09468236-0.08485920*I; b[5][2] =  0.4353193+0.0010994*I;
  
  b[0][3] = -0.0239881-0.4151801*I; b[0][4] = -0.6263347-0.5963434*I; b[0][5] = -0.45655201-0.02202738*I;
  b[1][3] = -0.1350729-0.0418095*I; b[1][4] = -0.6033738+0.0647601*I; b[1][5] = -0.28037632+0.30025691*I;
  b[2][3] = -0.1431319+0.0244497*I; b[2][4] = -0.2807683-0.0808173*I; b[2][5] =  0.12654249+0.21884983*I;
  b[3][3] =  0.2140318-0.4344302*I; b[3][4] = -0.1638382+0.0162849*I; b[3][5] = -0.17682708-0.12990665*I;
  b[4][3] = -0.4013470+0.0988086*I; b[4][4] = -0.3337646+0.9573819*I; b[4][5] =  0.28730090+0.30454484*I;
  b[5][3] = -0.1739908+0.0800473*I; b[5][4] = -0.2584657+0.3703075*I; b[5][5] =  0.09579707+0.08151071*I;
  
  six_invert(a);
  test = 0;

  for(int i = 0; i < 6; i++) {
    for(int j = 0; j < 6; j++) {
      if(creal(a[i][j] - b[i][j]) > EPS || cimag(a[i][j] - b[i][j]) > EPS) {
	printf("%d %d %e %e %e %e\n", i, j, creal(a[i][j]), cimag(a[i][j]), creal(b[i][j]), cimag(b[i][j]));
	test = 1;
      }
    }
  }

  assertFalseM(test,"The six_invert function does not work correctly!\n");
}