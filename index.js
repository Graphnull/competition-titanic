let csvParse = require('csv-parse/lib/sync')
let fs = require('fs')
let train= csvParse(fs.readFileSync('./train.csv'))
//let test_raw= csvParse(fs.readFileSync('./test.csv'))

train = train.map(v=>{
    //v[3]=v[3].slice(0,1)
    v[9]=(parseFloat(v[9])/10)|0
    return v;
})
let test = train.slice(train.length-100)
test.unshift(train[0])
train=train.slice(0,train.length-100)
let getAllocation = (field)=>{
    let index =train[0].findIndex(v=>v===field)
    let sum={}
    let count={}
    train.slice(1).forEach(data=>{
        if(!sum[data[index]]){
            sum[data[index]]=0;
            count[data[index]]=0;
        }
        sum[data[index]]+=parseInt(data[1])
        count[data[index]]++
    })
    let out ={}
    Object.keys(sum).map(field=>{
        out[field]={sum:sum[field],count:count[field],'res':sum[field]/count[field]}
    })
}
getAllocation('Pclass')
getAllocation('Age')
getAllocation('Sex')
getAllocation('Fare')
getAllocation('SibSp')
getAllocation('Parch')

let tf = require('@tensorflow/tfjs-node')
const model = tf.sequential();
model.add(tf.layers.dense(
    {units: 20, activation: 'elu', inputShape: [12]}));
model.add(tf.layers.dense({units: 40, activation: 'elu'}));
model.add(tf.layers.dense({units: 80, activation: 'elu'}));
model.add(tf.layers.dense({units: 1, activation: 'relu6'}));
model.summary();

const optimizer = tf.train.sgd(0.0001);
model.compile({
  optimizer,
  loss: tf.losses.meanSquaredError,
  metrics: ['accuracy'],
});
let convertXFunc = v=>{
       let pclass = [-1,-1,-1];
       pclass[v[2]-1]=1
       let Embarked =[-1,-1,-1,-1]
       if(v[11]==='C'){
        Embarked[0]=1
       }else if(v[11]==='Q'){
        Embarked[1]=1
       }else if(v[11]==='S'){
        Embarked[2]=1
       }else{
        Embarked[3]=1
       }
    let out =  pclass.concat([v[4]==='male'?1:-1,v[5]/70,v[9]/20])
    out=out.concat(Embarked)
    out.push(v[6]+1)
    out.push(v[7]+1)
    return out;
    //return [v[2]*0.3,v[4]==='male'?1:-1,v[5]/70,v[9]/20]
}
let convertYFunc = v=>{
    return [parseInt(v[1])*6]
}
let dataset = {
    x:tf.tensor(train.slice(1).map(convertXFunc)),
    y:tf.tensor(train.slice(1).map(convertYFunc))
}
let validationDataX = test.slice(1).map(convertXFunc)
let validationDataY = test.slice(1).map(convertYFunc)
let validation = {
    x:tf.tensor(validationDataX),
    y:tf.tensor(validationDataY)
}
let iter = 0
let bestScore = Infinity;
let bestModel = null;
let getHit = (model)=>{
    let count = test.length-1
    let hit=0;
    validationDataX.forEach((v,i)=>{
      let res =model.predict(tf.tensor([v])).dataSync()
     
      if(Math.round(res[0]/6)===Math.round(validationDataY[i]/6)){// console.log(res,val.slice(i*3,i*3+3) );
          hit++
      }
    })
 
      console.log('res: ', hit,count,hit/count*100,'%');
}
model.fit(dataset.x, dataset.y, {
    epochs: 1300,
    validationData: [validation.x, validation.y],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if(logs.val_loss<bestScore){
            bestScore=logs.val_loss
        await model.save(tf.io.withSaveHandler(artifacts => {
            bestModel = artifacts;
          }))//await tf.io.withSaveHandler(model)
        }
        iter++;
        if(iter%30===0){
            getHit(model)
        
          console.log(epoch,logs );
    }
            
      },
    }
  }).then(async (data)=>{
    let test_raw= csvParse(fs.readFileSync('./test.csv'))

    
    test_raw[0].unshift('PassengerId')
    test_raw[0][1]='Survived'
    let test= [test_raw[0]].concat(test_raw.slice(1).map(v=>{
        v.unshift(v[0])
        return v;
    }))
    let validationDataX = test.slice(1).map(convertXFunc)
      let loadedFromMemory = tf.io.fromMemory(bestModel)
      let pretreaned = await tf.loadLayersModel(loadedFromMemory)
     
   let csvsave = require('csv/lib/index')
    let result = [['PassengerId','Survived']]
   validationDataX.forEach((v,i)=>{
     let res =pretreaned.predict(tf.tensor([v])).dataSync()
    //console.log(test.slice(1)[i][0],res[0]/6);
     result.push([test.slice(1)[i][0],''+Math.round(res[0]/6)])


   })
   
      getHit(pretreaned)
   csvsave.stringify(result,(err, data)=>{
    fs.writeFileSync('./result.csv', data)
})
  }) 

