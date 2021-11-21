import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torchinfo as torchinfo
import pysbd
import pdb

from utils import *
from bidict import bidict, inverted
from tqdm import tqdm, trange
from argparser import args, DEVICE, CPU
from Datagenerator import DataGenerator
from Data import FakeNewsDataProcesser
from FDModel import *

DEBUG = False

class FraudDectionInterface(object):

    def __init__(self, model):
        file = './data/fake-news/train.csv'
        self.dp = FakeNewsDataProcesser()
        self.dp.read(file)
        feature, label = self.dp.default_process(split_dataset=True, vali_set=True)
        data_generator = DataGenerator(feature, label)
        self.model = FDModel(32, 768)
        self.trainer = FraudDectionTrainer(model, data_generator, args)
        self.trainer.load_model()
    
    def predict(self, author, title, text):
        self.dp.data = pd.DataFrame({"id": [0], "author": [author], "title": [title], "text": [text]})
        feature_test = self.dp.default_process(eval=True)
        pred = self.trainer.deploy_interface(feature_test)
        return float(pred.view(-1)[0])

class FraudDectionTrainer(object):

    def __init__(self, model:FDModel, data_generator:DataGenerator, args):
        self.model = model.to(TRAIN_DEVICE)
        self.data_generator = data_generator
        self.author_embedding = self.data_generator.author_embedding

        self.args = args

        self.batch_size = self.args.batch_size
        self.loss_type = self.args.loss_type
        self.epoch = self.args.epoch
        self.learning_rate = self.args.learning_rate

        self.optim = optim.Adam(list(self.model.parameters()) + list(self.author_embedding.parameters()), lr=0.01)
        self.loss = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        with trange(self.epoch) as progress:
            author_emb, title_emb, text_emb, label = self.data_generator.get_train_features()
            t_author_emb, t_title_emb, t_text_emb, t_label = self.data_generator.get_test_features()
            label = label.to(TRAIN_DEVICE)
            t_label = t_label.to(TRAIN_DEVICE)
            for ep in progress:
                try:
                    o = self.model(author_emb, title_emb, text_emb)
                    loss = self.loss(o, label)
                    out = torch.argmax(o.detach(), dim=1)
                    acc = (out.shape[0] - torch.count_nonzero(torch.logical_xor(out, label.detach()))) / out.shape[0]
                    loss.backward(retain_graph=True)
                    self.optim.step()
                    self.optim.zero_grad()
                    vali_loss, vali_acc = self.eval_epoch(t_author_emb, t_title_emb, t_text_emb, t_label)
                    progress.set_description(f"epoch{ep} train loss: {loss}, acc: {acc}, vali loss: {vali_loss}, vali_acc: {vali_acc}")
                except StopIteration:
                    pass
        torch.save(self.model.state_dict(), self.args.save_dir)
    
    def deploy_interface(self, feature):
        with torch.no_grad():
            author_emb, title_emb, text_emb = self.data_generator.get_eval_features(feature)
            try:
                o = self.model(author_emb, title_emb, text_emb)
            except StopIteration:
                pass
            return o.cpu()

    
    def eval_epoch(self, author_emb, title_emb, text_emb, label):
        with torch.no_grad():
            label = label.to(TRAIN_DEVICE)
            try:
                o = self.model(author_emb, title_emb, text_emb)
                loss = self.loss(o, label)
                out = torch.argmax(o.detach(), dim=1)
                acc = (out.shape[0] - torch.count_nonzero(torch.logical_xor(out, label.detach()))) / out.shape[0]
            except StopIteration:
                pass
            return loss, acc
    
    
    def eval(self, feature):
        with torch.no_grad():
            author_emb, title_emb, text_emb = self.data_generator.get_eval_features(feature)
            try:
                o = self.model(author_emb, title_emb, text_emb)
                out = torch.argmax(o.detach(), dim=1)
            except StopIteration:
                pass
            return out.cpu()
    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir))
    
    def eval_on_vali(self):
        self.load_model()
        self.model.eval()
        total_loss = 0
        for batch_idx in range(0, (self.vali_feature.shape[0] // self.batch_size) + 1):
            start_idx = min((batch_idx + 1) * self.batch_size, self.vali_feature.shape[0])
            end_idx = min((batch_idx + 1) * self.batch_size, self.vali_feature.shape[0])

            batch = self.vali_feature[start_idx:end_idx].to(DEVICE)
            scores = self.model.forward(batch)
            batch_label = self.vali_label[start_idx:end_idx].to(DEVICE)
            loss = self.loss(scores, batch_label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            total_loss += loss.item()
        print(f"Evaluation complete, total loss: {total_loss}")

    def eval_on_test(self):
        self.model.eval()
        total_loss = 0
        for batch_idx in range(0, (self.test_feature.shape[0] // self.batch_size) + 1):
            start_idx = min((batch_idx + 1) * self.batch_size, self.test_feature.shape[0])
            end_idx = min((batch_idx + 1) * self.batch_size, self.test_feature.shape[0])

            batch = self.test_feature[start_idx:end_idx].to(DEVICE)
            scores = self.model.forward(batch)
            batch_label = self.test_label[start_idx:end_idx].to(DEVICE)
            loss = self.loss(scores, batch_label)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            total_loss += loss.item()
        print(f"valuation complete, total loss: {total_loss}")


def deploy_test():
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)
    test_file = './data/fake-news/test.csv'
    feature, label = dp.default_process(split_dataset=True, vali_set=True)
    data_generator = DataGenerator(feature, label)
    model = FDModel(32, 768)
    trainer = FraudDectionTrainer(model, data_generator, args)
    trainer.load_model()
    author = "Thomas Bach"
    title = "Peng Shuai: Chinese tennis star tells Olympic officials she is safe"
    text = """
    In a statement, the IOC said its President Thomas Bach had spoken to Ms Peng for 30 minutes.

    "[She] was doing fine, which was our main concern," the statement read.

    Ms Peng, 35, disappeared from the public eye for almost three weeks after she made sexual assault allegations against a senior Chinese minister.

    Her absence triggered widespread concern, with international sports stars and governments calling on China to provide proof that she was safe.

    "At the beginning of the 30-minute call, Peng Shuai thanked the IOC for its concern about her wellbeing," the statement from the organisation said.

    "She explained that she is safe and well, living at her home in Beijing, but would like to have her privacy respected at this time," it added.

    "She prefers to spend her time with friends and family right now," it said. "Nevertheless, she will continue to be involved in tennis."

    The IOC statement also included an image of the video call taking place, with Ms Peng seen smiling to the camera.

    The outcry over the tennis star's apparent disappearance from the public eye prompted Chinese state media to release a series of photographs and videos that appeared to show all was well.

    Earlier on Sunday, a state media journalist posted a video clip on Twitter showing Ms Peng smiling while standing with officials at a tennis tournament in Beijing.

    Reuters news agency reported that the event organisers also published photos of the player on the event's official WeChat page.

    But a spokesperson from the Women's Tennis Association (WTA) told Reuters the videos were "insufficient" evidence of her safety and did not address its concerns about Ms Peng.

    "While it is positive to see her, it remains unclear if she is free and able to make decisions and take actions on her own, without coercion or external interference," WTA chief Steve Simon said at the time.

    """


    fake_author = "PATRICK HOWLEY"
    fake_title = "Exposed: FBI Director James Comey’s Clinton Foundation Connection"
    fake_text = """
    WASHINGTON, D.C. — A review of FBI Director James Comey’s professional history and relationships shows that the Obama cabinet leader — now under fire for his handling of the investigation of Hillary Clinton — is deeply entrenched in the big-money cronyism culture of Washington, D.C. His personal and professional relationships — all undisclosed as he announced the Bureau would not prosecute Clinton — reinforce bipartisan concerns that he may have politicized the criminal probe.

These concerns focus on millions of dollars that Comey accepted from a Clinton Foundation defense contractor, Comey’s former membership on a Clinton Foundation corporate partner’s board, and his surprising financial relationship with his brother Peter Comey, who works at the law firm that does the Clinton Foundation’s taxes.

Lockheed Martin

When President Obama nominated Comey to become FBI director in 2013, Comey promised the United States Senate that he would recuse himself on all cases involving former employers.

But Comey earned $6 million in one year alone from Lockheed Martin. Lockheed Martin became a Clinton Foundation donor that very year.

Comey served as deputy attorney general under John Ashcroft for two years of the Bush administration. When he left the Bush administration, he went directly to Lockheed Martin and became vice president, acting as a general counsel.

How much money did James Comey make from Lockheed Martin in his last year with the company, which he left in 2010? More than $6 million in compensation.

Lockheed Martin is a Clinton Foundation donor. The company admitted to becoming a Clinton Global Initiative member in 2010.

According to records, Lockheed Martin is also a member of the American Chamber of Commerce in Egypt, which paid Bill Clinton $250,000 to deliver a speech in 2010.

In 2010, Lockheed Martin won 17 approvals for private contracts from the Hillary Clinton State Department.

HSBC Holdings

In 2013, Comey became a board member, a director, and a Financial System Vulnerabilities Committee member of the London bank HSBC Holdings.

“Mr. Comey’s appointment will be for an initial three-year term which, subject to re-election by shareholders, will expire at the conclusion of the 2016 Annual General Meeting,” according to HSBC company records.

HSBC Holdings and its various philanthropic branches routinely partner with the Clinton Foundation. For instance, HSBC Holdings has partnered with Deutsche Bank through the Clinton Foundation to “retrofit 1,500 to 2,500 housing units, primarily in the low- to moderate-income sector” in “New York City.”

“Retrofitting” refers to a Green initiative to conserve energy in commercial housing units. Clinton Foundation records show that the Foundation projected “$1 billion in financing” for this Green initiative to conserve people’s energy in low-income housing units.

Who Is Peter Comey?

When our source called the Chinatown offices of D.C. law firm DLA Piper and asked for “Peter Comey,” a receptionist immediately put him through to Comey’s direct line. But Peter Comey is not featured on the DLA Piper website.

Peter Comey serves as “Senior Director of Real Estate Operations for the Americas” for DLA Piper. James Comey was not questioned about his relationship with Peter Comey in his confirmation hearing.

DLA Piper is the firm that performed the independent audit of the Clinton Foundation in November during Clinton-World’s first big push to put the email scandal behind them. DLA Piper’s employees taken as a whole represent a major Hillary Clinton 2016 campaign donation bloc and Clinton Foundation donation base.

DLA Piper ranks #5 on Hillary Clinton’s all-time career Top Contributors list, just ahead of Goldman Sachs.

And here is another thing: Peter Comey has a mortgage on his house that is owned by his brother James Comey, the FBI director.

Peter Comey’s financial records, obtained by Breitbart News, show that he bought a $950,000 house in Vienna, Virginia, in June 2008. He needed a $712,500 mortgage from First Savings Mortgage Corporation.

But on January 31, 2011, James Comey and his wife stepped in to become Private Party lenders. They granted a mortgage on the house for $711,000. Financial records suggest that Peter Comey took out two such mortgages from his brother that day.

This financial relationship between the Comey brothers began prior to James Comey’s nomination to become director of the FBI.

DLA Piper did not answer Breitbart News’ question as to whether James Comey and Peter Comey spoke at any point about this mortgage or anything else during the Clinton email investigation.

Peter Comey Re-Designed the FBI Building

FBI Director James Comey grew up in the New Jersey suburbs with his brother Peter. Both Comeys were briefly taken captive in 1977 by the “Ramsey rapist,” but the boys managed to escape through a window in their home, and neither boy was harmed.

James Comey became a prosecutor who worked on the Gambino crime family case. He went on to the Bush administration, a handful of private sector jobs, and then the Obama administration in 2013.

Peter Comey, meanwhile, went into construction.

After getting an MBA in real estate and urban development from George Washington University in 1998, Peter Comey became an executive at a company that re-designed George Washington University between 2004 and 2007 while his brother was in town working for the Bush administration.

In January 2009, at the beginning of the Obama administration, Peter Comey became “a real estate and construction consultant” for Procon Consulting.

Procon Consulting’s client list includes “FBI Headquarters Washington, DC.”

So what did Procon Consulting do for FBI Headquarters? Quite a bit, apparently. According to the firm’s records:

Procon provided strategic project management for the consolidation of over 11,000 FBI personnel into one, high security, facility.

Since 1972 the Federal Bureau of Investigation has had its headquarters in a purpose built 2.1 million square foot building on Pennsylvania Avenue. Having become functionally obsolete and in need of major repairs, GSA and the FBI were considering ways to meet the space needs required to maintain the Bureau’s mission and consolidate over 11,000 personnel.

Procon assisted GSA in assessing the FBI’s space needs and options for fulfilling those needs. Services provided included project management related to site evaluations, budgeting, due diligence, and the development of procurement and funding strategies.

Those “funding strategies” included talking to “stakeholders”: “Worked with stakeholders and key leadership to identify strategic objectives, goals and long range plans for capital and real estate projects.”

Procon Consulting obtained its contract for FBI Headquarters prior to James Comey’s nomination to serve as director of the FBI.

In June 2011, Peter Comey left Procon Consulting to become “Senior Director of Real Estate Operations for the Americas” for DLA Piper.

Peter Comey has generated some controversy in that role. According to Law360 in May 2013 (the same month that James Comey was confirmed as someone being considered by Obama to become FBI director):

Two real estate services businesses filed a $10 million suit against the law firm Monday alleging it stiffed them on as much as $760,000 of work done at DLA Piper’s Chicago office and improperly gave proprietary information to a competitor.

….

The plaintiffs take particular aim at Peter Comey, DLA Piper’s senior director of real estate operations. Leasecorp and SpaceLogik include several emails in the complaint that are purportedly from DLA Piper senior real estate partners Jay Epstein and Rich Klawiter and are sharply critical of Comey’s handling of the matter. In one email, Epstein wrote that “it’s an embarrassment for the firm to be treating someone who we are working with like this.”

In another email allegedly from Klawiter on Feb. 20, the DLA Piper partner informed Leasecorp President Michael Walker, a principal for both plaintiffs, that Comey had sent him and Epstein an email claiming that the real estate services firms were behind on their contractual obligations.

“I just received an email from Peter (Jay was also a recipient) that is so inflammatory I can’t even send it or you’ll hit the roof,” Klawiter said in the email, according to the complaint. “This is not going to end well.”
    """
    dp.data = pd.DataFrame({"id": [0], "author": [fake_author], "title": [fake_title], "text": [fake_text]})
    feature_test = dp.default_process(eval=True)
    pred = trainer.deploy_interface(feature_test)
    print(float(pred.view(-1)[0]))

def test():
    
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)
    test_file = './data/fake-news/test.csv'
    feature, label = dp.default_process(split_dataset=True, vali_set=True)
    data_generator = DataGenerator(feature, label)
    model = FDModel(32, 768)
    trainer = FraudDectionTrainer(model, data_generator, args)
    trainer.load_model()
    dp.read(test_file)
    feature_test = dp.default_process(eval=True)
    pred = trainer.eval(feature_test)
    with open('pred.csv', 'w') as f:
        f.write(f'id,label\n')
        for idx, uid in enumerate(dp.data['id']):
            f.write(f'{uid},{pred[idx]}\n')

def train():
    file = './data/fake-news/train.csv'
    dp = FakeNewsDataProcesser()
    dp.read(file)
    feature, label = dp.default_process(split_dataset=True, vali_set=True)
    data_generator = DataGenerator(feature, label)
    model = FDModel(32, 768)
    trainer = FraudDectionTrainer(model, data_generator, args)
    trainer.train()
  
def postprocess():
    dp = FakeNewsDataProcesser()
    test_file = './data/fake-news/test.csv'
    dp.read(test_file)
    import pandas as pd
    pred = pd.read_csv('pred.csv')
    l = dp.raw.merge(pred,how='left')
    l.fillna(1, inplace=True)
    m = l[['id', 'label']]
    m.to_csv('submit.csv', index=False)
if __name__ == "__main__":
    deploy_test()
    # postprocess()
    
    # se = SentenceEmbedding()
    # se.get_sentence_embeddings(None)
