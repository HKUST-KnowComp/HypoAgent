case_1p = {
  "answers": [6668,525,4882,6546,5652,1277,3360,2217,4138,5294,1838,4272,2489,3770,7099,5821,3904,68,7245,2391,344,6621,4197,486,3826,6005,1401,3453],
  "query": ["(","p","(",-5,")","(","e","(",4158,")",")",")"],
  "pattern_str": "(p,(e))",
  "query_nl": "Entities that have a 'CC' link to metoclopramide",
  "answers_nl": ["thiothixene","asenapine","paliperidone","tapentadol","rasagiline","chlorprothixene","iloperidone","ergonovine","methotrimeprazine","posaconazole","desvenlafaxine","molindone","fluspirilene","levomilnacipran","vilazodone","rotigotine","lurasidone","acepromazine","zuclopenthixol","fencamfamine","amisulpride","tetrabenazine","milnacipran","aripiprazole","lithium","sertindole","clomipramine","isocarboxazid"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "p e",
    "entitynumber": "1e",
    "relationnumber": "1p",
    "entity": "metoclopramide",
    "entity_id": "4158",
    "relation": "CC",
    "relation_id": "-5"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"p e\" and has 1 relation."
}

case_2p = {
  "answers": [4637,5442,966,4425,2569,5100,2254,1969,3826,1333,157,3383,4280,5754,7197,4606,5951],
  "query": ["(","p","(",-12,")","(","p","(",-7,")","(","e","(",5324,")",")",")",")"],
  "pattern_str": "(p,(p,(e)))",
  "query_nl": "Entities that have a 'K' link to an entity that has a 'E' link to ppox",
  "answers_nl": ["norepinephrine","propranolol","captopril","naloxone","furosemide","phenol","ethanol","dopamine","lithium","cimetidine","adenosine","indomethacin","morphine","riboflavin","yohimbine","nitric oxide","scopolamine"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "p p e",
    "entitynumber": "1e",
    "relationnumber": "2p",
    "entity": "ppox",
    "entity_id": "5324",
    "relation": "K",
    "relation_id": "-12"
  },
  "followup_question": "I want a hypothesis that includes the relation \"K\" and follows the pattern \"p p e\"."
}



case_2i ={
  "answers": [1476,5707,3596,781,614,4521,5037,3827,3828,5748,4087,3386,5563,5564,5501,3390],
  "query": ["(","i","(","p","(",-17,")","(","e","(",6718,")",")",")","(","p","(",-17,")","(","e","(",1070,")",")",")",")"],
  "pattern_str": "(i,(p,(e)),(p,(e)))",
  "query_nl": "Entities that have a 'P' link to tlr4, and have a 'P' link to cd2",
  "answers_nl": ["colitis","respiratory hypersensitivity","kidney diseases","body weight changes","autoimmune diseases","neoplasms","periodontitis","liver cirrhosis","liver cirrhosis biliary","rhinitis","melanoma","infections","purpura","purpura thrombocytopenic idiopathic","psoriasis","inflammation"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i p e p e",
    "entitynumber": "2e",
    "relationnumber": "2p",
    "entity": "tlr4",
    "entity_id": "6718",
    "relation": "P",
    "relation_id": "-17"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"i p e p e\" and has 2 relations."
}

case_3i ={
  "answers": [5056,5057,5058,5061,5062,5063,5053,5055],
  "query": ["(","i","(","i","(","p","(",-8,")","(","e","(",33,")",")",")","(","p","(",-3,")","(","e","(",5059,")",")",")",")","(","p","(",-3,")","(","e","(",5059,")",")",")",")"],
  "pattern_str": "(i,(i,(p,(e)),(p,(e))),(p,(e)))",
  "query_nl": "Entities that have a 'GG' link to abcd1, and have a 'B' link to pex19, and have a 'B' link to pex19",
  "answers_nl": ["pex13","pex14","pex16","pex3","pex5","pex6","pex10","pex12"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i i p e p e p e",
    "entitynumber": "3e",
    "relationnumber": "3p",
    "entity": "abcd1",
    "entity_id": "33",
    "relation": "GG",
    "relation_id": "-8"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"i i p e p e p e\" and has 3 relations."
}

case_ip ={
  "answers": [5644,1302,1303,1304,1305,1306,1307,1308,1309,1311,1312,1955,2597,2598,2599,2600,2602,4780,2605,2610,3130,5692,4568,3165,3166,3167,3174],
  "query": ["(","p","(",-20,")","(","i","(","p","(",-8,")","(","e","(",1305,")",")",")","(","p","(",-8,")","(","e","(",1312,")",")",")",")",")"],
  "pattern_str": "(p,(i,(p,(e)),(p,(e))))",
  "query_nl": "Entities that have a 'Ra' link to an entity that has a 'GG' link to chrna4, and has a 'GG' link to chrnb3",
  "answers_nl": ["rapsn","chrna10","chrna2","chrna3","chrna4","chrna5","chrna6","chrna7","chrna9","chrnb2","chrnb3","dnmt3b","gabra4","gabra5","gabra6","gabrb1","gabrb3","oprm1","gabrg2","gabrr2","hrh1","rela","nfatc4","htr1a","htr1b","htr1d","htr3b"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "p i p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "chrna4",
    "entity_id": "1305",
    "relation": "Ra",
    "relation_id": "-20"
  },
  "followup_question": "I want a hypothesis that contains 2 entities and has 3 relations."
}


case_pi = {
  "answers": [1217,2436,3910,3596,1868,6677,6679,4521,4395,6443,3827,4532,5363,7029,3829,3385,3386,3390],
  "query": ["(","i","(","p","(",-17,")","(","e","(",5222,")",")",")","(","p","(",-22,")","(","p","(",-5,")","(","e","(",5046,")",")",")",")",")"],
  "pattern_str": "(i,(p,(e)),(p,(p,(e))))",
  "query_nl": "Entities that have a 'P' link to plg, and have a 'Sa' link to an entity that has a 'CC' link to perphenazine",
  "answers_nl": ["cerebrovascular disorders","fibrosis","lymphoma","kidney diseases","diabetes mellitus","thromboembolism","thrombosis","neoplasms","myocardial ischemia","stroke","liver cirrhosis","nephritis","pre eclampsia","urologic neoplasms","liver diseases","infarction","infections","inflammation"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i p e p p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "plg",
    "entity_id": "5222",
    "relation": "P",
    "relation_id": "-17"
  },
  "followup_question": "I want a hypothesis that contains 2 entities and follows the pattern \"i p e p p e\"."
}


case_2u ={
  "answers": [1408,6335,1027,650,717,7054,2687,4753,6487,6491,3740,6492,4639,6689,353,5414,3943,635,2920,5804,5612,632,3768,5177,5178,4155,5758,5759],
  "query": ["(","u","(","p","(",-5,")","(","e","(",4364,")",")",")","(","p","(",-8,")","(","e","(",6485,")",")",")",")"],
  "pattern_str": "(u,(p,(e)),(p,(e)))",
  "query_nl": "Entities that either have a 'CC' link to mycophenolic acid or have a 'GG' link to sult1a1",
  "answers_nl": ["cloxacillin","sparfloxacin","cbr1","bacampicillin","benzylpenicillin","valganciclovir","gemifloxacin","ofloxacin","sult1a3","sult2a1","leflunomide","sult2b1","norfloxacin","ticarcillin","amoxicillin","probenecid","magnesium salicylate","azlocillin","gsto1","roflumilast","rabeprazole","azidocillin","levofloxacin","pivampicillin","pivmecillinam","meticillin","rifampicin","rifapentine"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "u p e p e",
    "entitynumber": "2e",
    "relationnumber": "2p",
    "entity": "mycophenolic acid",
    "entity_id": "4364",
    "relation": "CC",
    "relation_id": "-5"
  },
  "followup_question": "I want a hypothesis that contains the entity \"mycophenolic acid\" and contains 2 entities."
}

case_up = {
  "answers": [4606,1676,791,3362,5156,1193,6061,1839,1333,2242,7112,2761,2506,6604,2253,2767,6360,3673,6619,628,5366,4088,5883,3198],
  "query": ["(","p","(",-27,")","(","u","(","p","(",-26,")","(","e","(",3831,")",")",")","(","p","(",-8,")","(","e","(",4479,")",")",")",")",")"],
  "pattern_str": "(p,(u,(p,(e)),(p,(e))))",
  "query_nl": "Entities that have a 'Z' link to an entity that either has a 'X' link to liver neoplasms or has a 'GG' link to ndufb2",
  "answers_nl": ["nitric oxide","cyclosporine","bortezomib","imatinib","pioglitazone","celecoxib","simvastatin","dexamethasone","cimetidine","estradiol","vitamin c","glucosamine","folic acid","teniposide","ethambutol","glutathione","spironolactone","l carnitine","testosterone","azathioprine","prednisolone","melatonin","s adenosylmethionine","hydroxyproline"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "p u p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "liver neoplasms",
    "entity_id": "3831",
    "relation": "Z",
    "relation_id": "-27"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"p u p e p e\" and contains the entity \"liver neoplasms\"."
}

case_2in={
  "answers": [576,4481,4483,4484,4485,4487,5959,5960,4490,4493,6308,7016,7019,4718,4465,4466,4467,4468,4471,4472,4473,4474,4476,4477],
  "query": ["(","i","(","n","(","p","(",-8,")","(","e","(",1525,")",")",")",")","(","p","(",-8,")","(","e","(",4464,")",")",")",")"],
  "pattern_str": "(i,(n,(p,(e))),(p,(e)))",
  "query_nl": "Entities that do not have a 'GG' link to cox7b, and have a 'GG' link to ndufa1",
  "answers_nl": ["atp5a1","ndufb4","ndufb6","ndufb7","ndufb8","ndufc2","sdha","sdhb","ndufs3","ndufs6","sod1","uqcrc1","uqcrh","nubpl","ndufa10","ndufa13","ndufa2","ndufa3","ndufa5","ndufa6","ndufa7","ndufa8","ndufab1","ndufaf1"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p e p e",
    "entitynumber": "2e",
    "relationnumber": "2p",
    "entity": "cox7b",
    "entity_id": "1525",
    "relation": "GG",
    "relation_id": "-8"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"i n p e p e\" and includes the relation \"GG\"."
}

case_3in={
  "answers": [1654,1092,3334],
  "query": ["(","i","(","i","(","n","(","p","(",-13,")","(","e","(",987,")",")",")",")","(","p","(",-8,")","(","e","(",1055,")",")",")",")","(","p","(",-19,")","(","e","(",3469,")",")",")",")"],
  "pattern_str": "(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))",
  "query_nl": "Entities that do not have a 'ML' link to cardiovascular diseases, and have a 'GG' link to ccr1, and have a 'Q' link to itga2",
  "answers_nl": ["cxcr4","cd4","il2"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i i n p e p e p e",
    "entitynumber": "3e",
    "relationnumber": "3p",
    "entity": "cardiovascular diseases",
    "entity_id": "987",
    "relation": "ML",
    "relation_id": "-13"
  },
  "followup_question": "I want a hypothesis that contains the entity \"cardiovascular diseases\" and has 3 relations."
}


case_inp={
  "answers": [1696,3812,37,1221,4678,4682,13,2319,18,6739,20,1719,474,5214,991],
  "query": ["(","i","(","n","(","p","(",-3,")","(","p","(",-8,")","(","e","(",649,")",")",")",")",")","(","p","(",-21,")","(","e","(",19,")",")",")",")"],
  "pattern_str": "(i,(n,(p,(p,(e)))),(p,(e)))",
  "query_nl": "Entities that do not have a 'B' link to an entity that has a 'GG' link to baat, and have a 'Rg' link to abcb11",
  "answers_nl": ["cyp2a6","lipc","abcg5","ces1","nr0b2","nr1h4","abca1","fabp4","abcb1","tnf","abcb4","cyp7a1","arg1","pld2","carm1"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "baat",
    "entity_id": "649",
    "relation": "B",
    "relation_id": "-3"
  },
  "followup_question": "I want a hypothesis that includes the relation \"B\" and has 3 relations."
}

case_pni = {
  "answers": [5828,5001,5066,2941,5679,5456,2578,3891,2937,3546,6077,2463],
  "query": ["(","i","(","n","(","p","(",-8,")","(","p","(",-21,")","(","e","(",1128,")",")",")",")",")","(","p","(",-8,")","(","e","(",4922,")",")",")",")"],
  "pattern_str": "(i,(n,(p,(p,(e)))),(p,(e)))",
  "query_nl": "Entities that do not have a 'GG' link to an entity that has a 'Rg' link to cdh1, and have a 'GG' link to pask",
  "answers_nl": ["rpgrip1l","pdx1","pfkfb1","gys1","recql4","prpf6","fxn","ltk","gyg1","kcnh2","ski","flt4"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "cdh1",
    "entity_id": "1128",
    "relation": "GG",
    "relation_id": "-8"
  },
  "followup_question": "I want a hypothesis that includes the relation \"GG\" and contains 2 entities."
}

case_pin = {
  "answers": [4034,3528,907,2508,18,1432,1305,1150,1311,4068,4709,2535,2605,5936,5937,6450,5938,5940,3573,3574,5944,5243,702,3711],
  "query": ["(","i","(","n","(","p","(",-9,")","(","e","(",4159,")",")",")",")","(","p","(",-20,")","(","p","(",-20,")","(","e","(",3573,")",")",")",")",")"],
  "pattern_str": "(i,(n,(p,(e))),(p,(p,(e))))",
  "query_nl": "Entities that do not have a 'I' link to metocurine, and have a 'Ra' link to an entity that has a 'Ra' link to kcnq2",
  "answers_nl": ["mbd5","kcna1","cacna1h","folr1","abcb1","cntnap2","chrna4","cdkl5","chrnb2","mecp2","ntng1","foxg1","gabrg2","scn1a","scn1b","stxbp1","scn2a","scn3a","kcnq2","kcnq3","scn8a","pnkp","bdnf","lamc3"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p e p p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "metocurine",
    "entity_id": "4159",
    "relation": "I",
    "relation_id": "-9"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"i n p e p p e\" and has 3 relations."
}


case_3turn={
  "answers": [4608,4609,5377,5378,5091,3461,3462,711,1193,5099,7244,365,1271,4158,2490,3932,5598],
  "answers_nl": ["nitrofurantoin","nitroglycerin","prilocaine","primaquine","phenazopyridine","isosorbide dinitrate","isosorbide mononitrate","benzocaine","celecoxib","phenobarbital","zopiclone","amyl nitrite","chloroquine","metoclopramide","flutamide","mafenide","quinine"],
  "query": [["(","p","(",-5,")","(","u","(","p","(",-27,")","(","e","(",6645,")",")",")","(","p","(",-13,")","(","e","(",4521,")",")",")",")",")"],["(","p","(",-5,")","(","i","(","n","(","p","(",-5,")","(","e","(",922,")",")",")",")","(","p","(",-5,")","(","e","(",858,")",")",")",")",")"],["(","p","(",-5,")","(","u","(","p","(",-22,")","(","e","(",2185,")",")",")","(","p","(",-12,")","(","e","(",4898,")",")",")",")",")"]],
  "turn_count": 3,
  "turns": [
    {
      "turn_id": 1,
      "system_query_nl": "Entities that have a 'CC' link to an entity that either has a 'Z' link to tgm1 or has a 'ML' link to neoplasms",
      "system_pattern_str": "(p,(u,(p,(e)),(p,(e))))",
      "intention_mode": "two-condition",
      "followup_question": "I want a hypothesis that includes the relation \"CC\" and follows the pattern \"p u p e p e\".",
      "followup_condition_kinds": [
        "relation",
        "pattern"
      ],
      "followup_condition_values": {
        "pattern": "p u p e p e",
        "entitynumber": "2e",
        "relationnumber": "3p",
        "entity": "tgm1",
        "entity_id": "6645",
        "relation": "CC",
        "relation_id": "-5"
      }
    },
    {
      "turn_id": 2,
      "system_query_nl": "Entities that have a 'CC' link to an entity that does not have a 'CC' link to caffeine, and has a 'CC' link to butalbital",
      "system_pattern_str": "(p,(i,(n,(p,(e))),(p,(e))))",
      "intention_mode": "two-condition",
      "followup_question": "I want a hypothesis that includes the relation \"CC\" and contains the entity \"caffeine\".",
      "followup_condition_kinds": [
        "relation",
        "entity"
      ],
      "followup_condition_values": {
        "pattern": "p i n p e p e",
        "entitynumber": "2e",
        "relationnumber": "3p",
        "entity": "caffeine",
        "entity_id": "922",
        "relation": "CC",
        "relation_id": "-5"
      }
    },
    {
      "turn_id": 3,
      "system_query_nl": "Entities that have a 'CC' link to an entity that either has a 'Sa' link to epinephrine or has a 'K' link to panx1",
      "system_pattern_str": "(p,(u,(p,(e)),(p,(e))))",
      "intention_mode": "two-condition",
      "followup_question": "I want a hypothesis that contains 2 entities and includes the relation \"CC\".",
      "followup_condition_kinds": [
        "entitynumber",
        "relation"
      ],
      "followup_condition_values": {
        "pattern": "p u p e p e",
        "entitynumber": "2e",
        "relationnumber": "3p",
        "entity": "epinephrine",
        "entity_id": "2185",
        "relation": "CC",
        "relation_id": "-5"
      }
    }
  ]
}

case_complex = {
  "answers": [5315,5572,5317,5316,3749,3747,3748,6218,5318,5320,5319,5425,2196,3959,4857,4858,5571,6110],
  "answers_nl": ["ppia","pycrl","ppic","ppib","leprel2","lepre1","leprel1","slc6a14","ppif","ppih","ppig","prodh","eprs","malnutrition","p4ha1","p4ha2","pycr1","slc16a10"],
  "query": [["(","p","(",-9,")","(","p","(",-9,")","(","e","(",5315,")",")",")",")"],["(","p","(",-9,")","(","u","(","p","(",-17,")","(","e","(",258,")",")",")","(","p","(",-9,")","(","e","(",5315,")",")",")",")",")"],["(","p","(",-9,")","(","p","(",-27,")","(","e","(",6187,")",")",")",")"]],
  "turn_count": 3,
  "turns": [
    {
      "turn_id": 1,
      "system_query_nl": "Entities that have a 'I' link to an entity that has a 'I' link to ppia",
      "system_pattern_str": "(p,(p,(e)))",
      "intention_mode": "two-condition",
      "followup_question": "I want a hypothesis that contains the entity \"ppia\" and contains 1 entity.",
      "followup_condition_kinds": [
        "entitynumber",
        "entity"
      ],
      "followup_condition_values": {
        "pattern": "p p e",
        "entitynumber": "1e",
        "relationnumber": "2p",
        "entity": "ppia",
        "entity_id": "5315",
        "relation": "I",
        "relation_id": "-9"
      }
    },
    {
      "turn_id": 2,
      "system_query_nl": "Entities that have a 'I' link to an entity that either has a 'P' link to aldh1a1 or has a 'I' link to ppia",
      "system_pattern_str": "(p,(u,(p,(e)),(p,(e))))",
      "intention_mode": "two-condition",
      "followup_question": "I want a hypothesis that contains 2 entities and contains the entity \"aldh1a1\".",
      "followup_condition_kinds": [
        "entitynumber",
        "entity"
      ],
      "followup_condition_values": {
        "pattern": "p u p e p e",
        "entitynumber": "2e",
        "relationnumber": "3p",
        "entity": "aldh1a1",
        "entity_id": "258",
        "relation": "I",
        "relation_id": "-9"
      }
    },
    {
      "turn_id": 3,
      "system_query_nl": "Entities that have a 'I' link to an entity that has a 'Z' link to slc36a1",
      "system_pattern_str": "(p,(p,(e)))",
      "intention_mode": "simplify-logic",
      "followup_question": "This is too complex. I want to make the logic simpler."
    }
  ]
}