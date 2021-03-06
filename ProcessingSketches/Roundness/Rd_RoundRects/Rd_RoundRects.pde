int cnt = 0;

void setup() {
  size(224, 224);
  noLoop();
}

void randrect(float rad) {  
  
  float x = random(0, width);
  float y = random(0, height);
  float w = random(width/8, width/2);
  float h = random(height/8, height/2);
  float a = random(PI);
  float m = min(w, h) / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, h, rad * m);
  popMatrix();
}

void draw() {
  for (int iteration = 100; iteration < 166; iteration++) {
    for (int r=0; r<10; r++) {
      for (int blur = 0; blur <= 2; blur++) {

        clear();
        float back_col =random(0, 255);
        background(back_col);
        for (int i=0; i<random(2, 25); i++) {
          float col = random(0, 255);
          while ( col < back_col+70 && col > back_col-70) {
            col = random(0, 255);
          }
          noStroke();
          fill(col);
          
          if (r!=9) {
          float rad = r/9.0;
          randrect(rad);
          
          } 
          else if (r==9) {
            float radius = random(width/9, width/3);
            float x = random(20, width-20);
            float y = random(20, height-20);       
            ellipse(x, y, radius, radius);
          }
        
        }
        
        float b = blur;
        if (blur==2) {
          b = 1.5;
        }
        PImage temp = get();
        temp.filter(BLUR, b);

        String fname = String.format("Output/tri-r%02d-rr%03db%01d.png", r, iteration, blur);
        temp.save(fname);
      }
    }
  }
}