color b1, b2;
float smt;

void setup() {
  size(318, 318);
  ellipseMode(CENTER);
  noLoop();
}

void draw() {
  for (int iteration =0; iteration<300; iteration ++) {
    clear();
    int r1 = int(random(0,50));
    int r2 = int(random(205,255));
    b1 = color(r1);
    b2 = color(r2);
    for (int level=0; level<10; level++) {
      if (level <= 6) {
        smt = map(level, 0, 7, 0, 0.2);
      } else if (level>= 7) {
        smt = map(level, 7, 9, 0.25, 1);
      }
      //print("level:",level,"smt: ",smt,"\n");
      conicGradient(r1, r2, smt);

      PImage current = get();
      float rn = random(0,1);
      float rad = map(rn, 0, 1, 0, TWO_PI);
      rotation(current, rad);
        
      PImage temp = get(47,47,224,224);
      String fileName = String.format("Output/tri-c%02d-cn%03d.png", level,iteration);
      temp.save(fileName);
    }
  }
}



void rotation(PImage img, float rad) {
  pushMatrix();
  translate(width/2, height/2);
  rotate(rad);
  translate(-width/2, -height/2);
  image(img, 0, 0);
  popMatrix();
}


void conicGradient(int r1, int r2, float smt) {
  
  //int r_start = int(random(0,360));
  
  //for(float r = r_start; r < r_start+360; r+=1){
    
  //  float rad = radians(r);
  //  float rad_e = radians(r+1);
  //  float noise = random(0,255);
  //  float inter = map(r, r_start, r_start+360, 0 ,1);
  //  color col = lerpColor(r1,r2,inter);
  //  //stroke(col*(1-smt)+smt*noise);
  //  fill(col*(1-smt)+smt*noise);
  //  arc(width/2,height/2,width,height,rad,rad_e);
    
  //}

  for (int x = 0; x<= width-2; x+=2) {
    float noise = random(0, 255);
    float inter = map(x, 0, 4*width, 0, 1);
    color col = lerpColor(r1, r2, inter);
    stroke(col*(1-smt)+smt*noise);
    fill(col*(1-smt)+smt*noise);
    //line(width/2,height/2,x,0);
    triangle(x, 0, x+2, 0, width/2, height/2);
  }

  for (int y = 0; y<= height-2; y+=2) {
    float noise = random(0, 255);
    float inter = map(y+width, 0, 4*width, 0, 1);
    color col = lerpColor(r1, r2, inter);
    stroke(col*(1-smt)+smt*noise);
    fill(col*(1-smt)+smt*noise);
    //line(width/2,height/2,x,0);
    triangle(width, y, width, y+2, width/2, height/2);
  }

  for (int x = 0; x<= width-2; x+=2) {
    float noise = random(0, 255);
    float inter = map(x+2*width, 0, 4*width, 0, 1);
    color col = lerpColor(r1, r2, inter);
    stroke(col*(1-smt)+smt*noise);
    fill(col*(1-smt)+smt*noise);
    triangle(width-x, height, (width-x)-2, height, width/2, height/2);
  }

  for (int y = 0; y<= height-2; y++) {
    float noise = random(0, 255);
    float inter = map(y+3*width, 0, 4*width, 0, 1);
    color col = lerpColor(r1, r2, inter);
    stroke(col*(1-smt)+smt*noise);
    fill(col*(1-smt)+smt*noise);
    //line(width/2,height/2,x,0);
    triangle(0, height-y, 0, (height-y)-2, width/2, height/2);
  }
}
