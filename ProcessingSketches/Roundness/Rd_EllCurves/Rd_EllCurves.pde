
void setup() {
  size(500, 500);
  noLoop();
  ellipseMode(RADIUS);
}


void draw() {

  for (int iteration = 0; iteration<25; iteration++) {
    for (int level = 9; level >= 0; level--) {
      for (int blur = 0; blur<=4; blur++) {
        for(int noise=0; noise<=3; noise++){    
          float l = map(level, 9, 0, 1, 0.01);

          clear();
          float back_col = random(0, 255);
          background(back_col);
          
          
          for (int x = 0; x<= 10; x++) {
            
           
            float stroke_col = random(0, 255);
            while (stroke_col<back_col+70 && stroke_col > back_col-70) {
              stroke_col = random(0, 255);
            }
            stroke(stroke_col);
            
            noFill();
            int rand1;
            if(blur<=2){
              rand1 = int(random(2,4));   
            }
            else{
              rand1 = int(random(5,10));
            }
            strokeWeight(rand1);

            float orig_x = 20;
            float orig_y = 20/l;
            float rand2 = random(0.5,1);
            ellipse(width/2, height/2, orig_x+50*x*rand2, orig_y+50*x*rand2);
            
          }
          
          PImage current = get();
          rotation(current);

          for (int point = 0; point < noise*50; point++) {
            strokeWeight(2);
            stroke(random(0, 50), 20);
            point(random(0, width), random(0, height));
          }
          int trans = int(random(-50,50));
          PImage temp = get(138+trans, 138+trans, 224, 224);
          temp.filter(BLUR,blur);
          String filename = String.format("Output/tri-r%02d-cv%03dn%01db%01d.png", level, iteration,noise, blur);
          temp.save(filename);
        }
      }
    }
  }
}


void rotation(PImage img) {
  float rad = random(0, TWO_PI);
  pushMatrix();
  translate(width/2, height/2);
  rotate(rad);
  translate(-width/2, -height/2);
  image(img, 0, 0);
  popMatrix();
}