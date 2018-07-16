int back_col;

void setup() {
  size(224, 224);
  noStroke();
  rectMode(RADIUS);
  ellipseMode(RADIUS);
  noLoop();
}

void draw() {

  for (int iteration=0; iteration<120; iteration++) {
    for (int r = 0; r<10; r++) {

      for (int n = 1; n <= 2; n++) {
        clear();
        back_col = int(random(0, 100));
        background(back_col);  
        drawGradientSet(n, back_col, float(r));
        PImage temp = get();
        String fileName = String.format("Output/tri-r%02d-rs%03d%01d.png", r, iteration, n);
        temp.save(fileName);
      }
    }
  }
}

void drawGradientSet(int num, int back_col, float rad) {
  for (int n = 1; n<= num; n++) {
    drawGradient(1.0, back_col, rad);
  }
}


void drawGradient(float smt, int back_col, float rad) {
  int radius = int(random(40, width/4));
  int rand = int(random(5,10));
  int x = int(random(10, width-10));
  int y = int(random(10, width-10));

  for (int r = radius+rand; r >= 0; r-=smt) {

    float inter = map(r, 0, radius+rand, 0, 1);
    float col = lerpColor(255, back_col, inter);
    fill(col);
    if(rad!=9){
      float factor = rad/9;
      rect(x, y, r+rand, r+rand, factor*r);
    }
    else if(rad==9){
     
      ellipse(x, y, r+rand, r+rand);
    }
  }
}
