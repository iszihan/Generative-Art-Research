int cnt = 0;
float back_col;

void setup() {
  size(224, 224);
  noLoop();
  rectMode(CENTER);
}

void randrect(float rad) {  
  float r = random(0, 255);
  while ( r < back_col+80 && r > back_col-80) {
    r = random(0, 255);
  }
  stroke(r);
  float wt = random(0, 10);
  strokeWeight(wt);
  noFill();
  float x = random(20, width-20);
  float y = random(20, height-20);
  float w = random(width/8, width/3);
  //float h = random(height/8, height/3);
  float a = random(PI);
  float m = w / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, w, rad * m);
  popMatrix();
}

void draw() {
  for ( int iteration = 0; iteration < 17; iteration++) {
    for (int round = 0; round < 10; round++) {
      for (int messy = 1; messy <= 10; messy++) {
        for (int blur = 0; blur <= 2; blur++) {
          clear();
          back_col = random(0, 255);
          background(back_col);
          if (round != 9) {

            float r_level = round*10;
            float m_level = messy*messy;
            float rad = r_level / 90.0;
            for (int i=0; i < Math.round((m_level+3)/2); i++) {
              randrect(rad);
            }
          } else if (round == 9) {
            float m_level = messy*messy;
            for (int i=0; i < Math.round((m_level+3)/2); i++) {
              float radius = random(width/9, width/3);
              float x = random(20, width-20);
              float y = random(20, height-20);
              float r = random(0, 255);
              while ( r < back_col+80 && r > back_col-80) {
                r = random(0, 255);
              }
              stroke(r);
              float wt = random(0, 10);
              strokeWeight(wt);
              noFill();

              ellipse(x, y, radius, radius);
            }
          }
          PImage temp = get();
          float b = blur;
          if (blur ==2 ) {
            b = 1.2;
          }
          temp.filter(BLUR, b);
          String fname = String.format("Output/tri-r%02d-m%02d-nf%02d%01d.png", round, messy, iteration, blur);
          temp.save(fname);
        }
      }
    }
  }
}
