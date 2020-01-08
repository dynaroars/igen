
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Note
import FindConfig
import Ifdefguards ( findIfdefPatterns )
import qualified LocSize as Loc

import Control.Monad ( forM )
import Data.ByteString ( ByteString )
import Data.List ( group, sort, zipWith3, zipWith5, zipWith6 )
import qualified Data.List.Ordered as OL

import System.Console.CmdArgs
import System.FilePath

import Debug.Trace

data VBDb
    = Check { db :: FilePath, project :: FilePath }
    | TypeFreq { db :: FilePath, project :: FilePath }
    | DegreeFreq { db :: FilePath, project :: FilePath }
    | ConfigFreq { db :: FilePath, project :: FilePath }
    | Config_Freq2 { db :: FilePath, project :: FilePath }
    | Config_Loc { db :: FilePath, project :: FilePath }
    | LayerFreq { db :: FilePath, project :: FilePath }
    | PcFreq { db :: FilePath, project :: FilePath }
    | Lst_Config { db :: FilePath, project :: FilePath }
    | Cnt_Config { db :: FilePath, project :: FilePath }
    | Tbl_Config { db :: FilePath, project :: FilePath }
    | LocCfgDist { db :: FilePath, project :: FilePath }
    | LocStats { src :: FilePath }
    | Loc_Freq { db :: FilePath, project :: FilePath }
    | Ifdef_Freq { db :: FilePath, project :: FilePath }
    deriving (Show, Data, Typeable)

vbdbCheck :: VBDb
vbdbCheck =
    Check {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Linter for bug reports"

vbdbTypeFreq :: VBDb
vbdbTypeFreq =
    TypeFreq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Frequency of bug types"
    &= groupname "Statistics"

vbdbDegreeFreq :: VBDb
vbdbDegreeFreq =
    DegreeFreq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Frequency of n-degree bugs"

vbdbConfigFreq :: VBDb
vbdbConfigFreq =
    ConfigFreq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Frequency of config-options"
    &= groupname "Statistics"

vbdbLayerFreq :: VBDb
vbdbLayerFreq =
    LayerFreq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Frequency of layer-fix"
    &= groupname "Statistics"

vbdbPcFreq :: VBDb
vbdbPcFreq =
    PcFreq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Frequency of presence-condition (PC)"
    &= groupname "Statistics"

vbdbLstConfig :: VBDb
vbdbLstConfig =
    Lst_Config {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "List of config-options"
    &= groupname "Statistics"

vbdbCntConfig :: VBDb
vbdbCntConfig =
    Cnt_Config {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Number of config-options"
    &= groupname "Statistics"

vbdbTblConfig :: VBDb
vbdbTblConfig =
    Tbl_Config {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "LaTeX table of config-options (3-columns)"
    &= groupname "LaTeX"

vbdbLocCfgDist :: VBDb
vbdbLocCfgDist =
    LocCfgDist {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= groupname "Statistics"

vbdbLocStats :: VBDb
vbdbLocStats =
    LocStats {
      src = "." &= typ "PATH" &= help "Path to project sources"
    }
    &= groupname "Statistics"

vbdbConfigFreq2 :: VBDb
vbdbConfigFreq2 =
    Config_Freq2 {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "How many configs occur N times?"
    &= groupname "Statistics"

vbdbConfigLoc :: VBDb
vbdbConfigLoc =
    Config_Loc {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "Where are configs defined?"
    &= groupname "Statistics"

vbdbLocFreq :: VBDb
vbdbLocFreq =
    Loc_Freq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "How many bugs in a given subsystem?"
    &= groupname "Statistics"

vbdbIfdefFreq :: VBDb
vbdbIfdefFreq =
    Ifdef_Freq {
      db = "." &= typ "PATH" &= help "Path to VBDb"
    , project = "linux" &= typ "FOLDER" &= help "Project folder"
    }
    &= help "What is the frequency of a given #ifdef form?"
    &= groupname "Statistics"

vbdb :: VBDb
vbdb = modes [vbdbCheck &= auto
             ,vbdbTypeFreq
             ,vbdbDegreeFreq
             ,vbdbConfigFreq
             ,vbdbLayerFreq
             ,vbdbPcFreq
             ,vbdbLstConfig
             ,vbdbCntConfig
             ,vbdbTblConfig
             ,vbdbLocCfgDist
             ,vbdbLocStats
             ,vbdbConfigFreq2
             ,vbdbConfigLoc
             ,vbdbLocFreq
             ,vbdbIfdefFreq
             ]
    &= summary "VBDb tool"

mkNotesPath :: FilePath -> FilePath -> FilePath
mkNotesPath db project = db </> project </> "report"

main :: IO ()
main = do
    args <- cmdArgs vbdb
    case args of
        Check{..} -> checkNotes db project
        TypeFreq{..} -> onNotesPrint (mkNotesPath db project) freqOfType
        DegreeFreq{..} -> onNotesPrint (mkNotesPath db project) freqOfDegree
        ConfigFreq{..} -> onNotesPrint (mkNotesPath db project) freqOfConfig
        LayerFreq{..}  -> onNotesPrint (mkNotesPath db project) freqOfFixin
        PcFreq{..}     -> onNotesPrint (mkNotesPath db project) freqOfPcps
        Lst_Config{..} -> onNotesM (mkNotesPath db project) $ mapM_ (putStrLn . pp) . listOfConfig
        Cnt_Config{..} -> onNotesPrint (mkNotesPath db project) numberOfConfig
        Tbl_Config{..} -> onNotesM (mkNotesPath db project) $ putStr . latexConfigTable3
--        "tex-cfg5" -> onNotesM $ putStrLn . latexFeatureTable5
--        "tex-cfg6" -> onNotesM $ putStrLn . latexFeatureTable6
        LocCfgDist{..} -> onNotesPrintM (mkNotesPath db project) locConfigDist
        LocStats{..}   -> putStr =<< Loc.showStats src
        Config_Freq2{..} -> onNotesPrint (mkNotesPath db project) freqOfFreqOfConfig
        Config_Loc{..} -> onNotesPrintM (mkNotesPath db project) configLoc
        Loc_Freq{..} -> onNotesPrint (mkNotesPath db project) freqOfLoc0
        -- TODO: Use onBenchsM
        Ifdef_Freq{..} -> putStrLn . pp =<< freqOfIfpats (db </> project </> "simple")

-- * LaTeX

latexConfigTable3 :: [Note] -> String
latexConfigTable3 ns = unlines $
   zipWith3 mkRow
      (map pp fs1)
      (fill $ map pp fs2)
      (fill $ map pp fs3)
  where mkRow :: String -> String -> String -> String
        mkRow a b c = a ++ " & " ++ b ++ " & " ++ c ++ "\\\\"
        fs :: [String]
        fs = map ppWithFreq $ freqOfConfig ns
        num = length fs
        rowsNo = ceiling(fromIntegral num / 3::Double)
        fill :: [String] -> [String]
        fill xs = xs ++ replicate dif ""
            where dif = rowsNo - length xs
        fs1, fs2, fs3 :: [String]
        (fs1,fs') = splitAt rowsNo fs
        (fs2,fs3) = splitAt rowsNo fs'

ppWithFreq :: PP a => (a,Int) -> String
ppWithFreq (x,1) = pp x
ppWithFreq (x,n) = pp x ++ " (x" ++ pp n ++ ")"

latexFeatureTable5 :: [Note] -> String
latexFeatureTable5 ns = unlines $
   zipWith5 mkRow
      (fill $ map pp fs1)
      (fill $ map pp fs2)
      (fill $ map pp fs3)
      (fill $ map pp fs4)
      (fill $ map pp fs5)
  where mkRow :: String -> String -> String -> String -> String -> String
        mkRow a b c d e = a ++ " & " ++ b ++ " & " ++ c ++ " & " ++ d ++ " & " ++ e ++ "\\\\"
        fs :: [Config]
        fs = map fst $ freqOfConfig ns
        num = length fs
        colLen = ceiling(fromIntegral num / 5::Double)
        fill :: [String] -> [String]
        fill xs | length xs < colLen = xs ++ [""]
                | otherwise          = xs
        fs1, fs2, fs3, fs4, fs5 :: [Config]
        (fs1,fs') = splitAt colLen fs
        (fs2,fs'') = splitAt colLen fs'
        (fs3,fs''') = splitAt colLen fs''
        (fs4,fs5) =  splitAt colLen fs'''

latexFeatureTable6 :: [Note] -> String
latexFeatureTable6 ns = unlines $
   zipWith6 mkRow
      (fill $ map pp fs1)
      (fill $ map pp fs2)
      (fill $ map pp fs3)
      (fill $ map pp fs4)
      (fill $ map pp fs5)
      (fill $ map pp fs6)
  where mkRow :: String -> String -> String -> String -> String -> String -> String
        mkRow a b c d e f = a ++ " & " ++ b ++ " & " ++ c ++ " & " ++ d ++ " & " ++ e ++ " & " ++ f ++ "\\\\"
        fs :: [Config]
        fs = map fst $ freqOfConfig ns
        num = length fs
        colLen = ceiling(fromIntegral num / 6::Double)
        fill :: [String] -> [String]
        fill xs | length xs < colLen = xs ++ [""]
                | otherwise          = xs
        fs1, fs2, fs3, fs4, fs5 :: [Config]
        (fs1,fs') = splitAt colLen fs
        (fs2,fs'') = splitAt colLen fs'
        (fs3,fs''') = splitAt colLen fs''
        (fs4,fs'''') = splitAt colLen fs'''
        (fs5,fs6) =  splitAt colLen fs''''

-- * Stats

freqOfType :: [Note] -> [(Type,Int)]
freqOfType = count . map noteType

bugDegree :: Note -> Int
bugDegree = length . noteConfigs

freqOfDegree :: [Note] -> [(Int,Int)]
freqOfDegree = count . map bugDegree

listOfConfig :: [Note] -> [Config]
listOfConfig = OL.nub . sort . concatMap noteConfigs

numberOfConfig :: [Note] -> Int
numberOfConfig = length . listOfConfig

freqOfConfig :: [Note] -> [(Config,Int)]
freqOfConfig = count . concatMap noteConfigs

freqOfPcps :: [Note] -> [(PcPat,Int)]
freqOfPcps ns = [ (pat,length pcs)
                | (pat,pcs) <- groupByPat configPats $ map notePc ns ]

freqOfFreqOfConfig :: [Note] -> [(Int,Int)]
freqOfFreqOfConfig = count . map snd . freqOfConfig

configLoc :: [Note] -> IO [(FilePath,Int)]
configLoc ns = do
  lss <- forM ns $ \n -> findConfigs (noteHash n) (noteConfigs n)
  let ls = concat lss
  return $ count ls

locConfigDist :: [Note] -> IO [(Int,Int)]
locConfigDist ns = do
  ds <- forM ns $ \n -> do
    ls <- findConfigs (noteHash n) (noteConfigs n)
    let nl0 = noteLoc0 n
        ls0 = map headPath $ stripQMarks ls
    return $ length $ filter (/= nl0) ls0
  return $ count ds
  where stripQMarks = map (dropWhile (== '?'))

freqOfLoc0 :: [Note] -> [(FilePath,Int)]
freqOfLoc0 = count . map noteLoc0

freqOfFixin :: [Note] -> [([FixedIn],Int)]
freqOfFixin = count . map noteFixed

freqOfIfpats :: FilePath -> IO [(PcPat,Int)]
freqOfIfpats benchsDir = do
  pcs <- findIfdefPatterns benchsDir
  return [ (pat,length pat_pcs) | (pat,pat_pcs) <- groupByPat ifdefPats pcs ]

-- * PC patterns

data PcPat = PcPat { pcpName :: !ByteString, pcpPred :: Pc -> Bool }
           | Other !Pc

instance PP PcPat where
  pp PcPat{pcpName}  = pp pcpName
  pp (Other pc)      = pp pc


configPats :: [PcPat]
configPats =
  [ PcPat "A" $ \pc -> case pc of
       [[Pos _]] -> True
       _         -> False
   , PcPat "A && B" $ \pc -> case pc of
       [[Pos _],[Pos _]] -> True
       _                 -> False
   , PcPat "A && B && C" $ \pc -> case pc of
       [[Pos _],[Pos _],[Pos _]] -> True
       _                         -> False
   , PcPat "A && B && C && D && E" $ \pc -> case pc of
       [[Pos _],[Pos _],[Pos _],[Pos _],[Pos _]] -> True
       _                                         -> False
   , PcPat "!A" $ \pc -> case pc of
       [[Neg _]] -> True
       _         -> False
   , PcPat "!A && !B" $ \pc -> case pc of
       [[Neg _],[Neg _]] -> True
       _                 -> False
   , PcPat "A && !B" $ \pc -> case pc of
       [[Pos _],[Neg _]] -> True
       [[Neg _],[Pos _]] -> True
       _                 -> False
   , PcPat "A && B && !C" $ \pc -> case pc of
       [[Pos _],[Pos _],[Neg _]] -> True
       _                         -> False
   , PcPat "A && B && C && !D" $ \pc -> case pc of
       [[Pos _],[Pos _],[Pos _],[Neg _]] -> True
       _                                 -> False
   , PcPat "A && B && C && D && !E" $ \pc -> case pc of
       [[Pos _],[Pos _],[Neg _],[Pos _],[Pos _]] -> True
       _                                         -> False
   , PcPat "A && !B && !C && !D && !E" $ \pc -> case pc of
       [[Neg _],[Pos _],[Neg _],[Neg _],[Neg _]] -> True
       _                                         -> False
   , PcPat "!A && (B || C)" $ \pc -> case pc of
       [[Pos _,Pos _],[Neg _]] -> True
       _                       -> False
   ]

pcpOther :: [PcPat] -> Pc -> Bool
pcpOther pcps pc = not $ or [ pcpPred pc | PcPat{pcpPred} <- pcps ]

groupByPat :: [PcPat] -> [Pc] -> [(PcPat,[Pc])]
groupByPat pcps pcs =
     [ (pcp,filter pcpPred pcs) | pcp@PcPat{pcpPred} <- pcps  ]
  ++ [ (Other pc,[pc])          | pc <- pcs, pcpOther pcps pc ]

-- * Ifdef guard patterns

ifdefPats :: [PcPat]
ifdefPats =
  [ PcPat "A" $ \pc -> case pc of
       [[Pos _]] -> True
       _         -> False
   , PcPat "A || B" $ \pc -> case pc of
       [[Pos _,Pos _]] -> True
       _               -> False
   , PcPat "A && B" $ \pc -> case pc of
       [[Pos _],[Pos _]] -> True
       _                 -> False
   , PcPat "A && B && C" $ \pc -> case pc of
       [[Pos _],[Pos _],[Pos _]] -> True
       _                         -> False
   , PcPat "!A" $ \pc -> case pc of
       [[Neg _]] -> True
       _         -> False
   , PcPat "!A && !B && !C" $ \pc -> case pc of
       [[Neg _],[Neg _],[Neg _]] -> True
       _                         -> False
   ]

-- * Utils

count :: Ord a => [a] -> [(a,Int)]
count xs = map head xss `zip` map length xss
  where xss = group $ sort xs

